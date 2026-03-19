#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Run this cmd to build: c++ -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup $(python3 -m pybind11 --includes) src/envs/soccer_envs/soccer_env.cpp -o soccer_sim$(python3-config --extension-suffix)

namespace py = pybind11;

enum EntityType { BALL, PLAYER_A, PLAYER_B };

struct Entity {
    float x, y, vx, vy, radius, mass;
    EntityType type;
    int id;
};

class SoccerEnv {
public:
    std::vector<Entity> entities;
    float width, height, dt;
    int num_team_a, num_team_b;
    int steps = 0;
    const int max_steps = 1000;
    const float kick_range = 2.0f;
    float top_speed = 5.0f;
    float accel = 0.5f;
    
    // Track previous state for reward calculation
    float prev_ball_dist_to_goal_b = 0;
    float prev_ball_dist_to_goal_a = 0;
    std::vector<float> prev_player_ball_distances;

    SoccerEnv(int num_a, int num_b, float w, float h, float time_step) 
        : width(w), height(h), dt(time_step), num_team_a(num_a), num_team_b(num_b) {
        reset();
    }

    void reset() {
        entities.clear();
        steps = 0;
        
        entities.push_back({width / 2, height / 2, 0, 0, 0.5f, 0.05f, BALL, 0});

        for (int i = 0; i < num_team_a; ++i) {
            float y = height / (num_team_a + 1) * (i + 1);
            entities.push_back({width / 4, y, 0, 0, 1.0f, 1.0f, PLAYER_A, i});
        }

        for (int i = 0; i < num_team_b; ++i) {
            float y = height / (num_team_b + 1) * (i + 1);
            entities.push_back({3 * width / 4, y, 0, 0, 1.0f, 1.0f, PLAYER_B, i});
        }
        
        update_distance_tracking();
    }

    std::vector<float> step(std::vector<int> actions) {
        apply_actions(actions);
        update_positions();
        handle_collisions();
        check_goals();
        steps++;

        return get_rewards();
    }

    std::vector<float> get_state() {
        std::vector<float> state;
        for (auto& entity : entities) {
            state.push_back(entity.x / width);
            state.push_back(entity.y / height);
            state.push_back(entity.vx / top_speed);
            state.push_back(entity.vy / top_speed);
        }
        return state;
    }

    bool is_done() {
        return steps >= max_steps;
    }

private:
    void apply_actions(std::vector<int>& actions) {
        for (size_t i = 1; i < entities.size(); ++i) {
            if (i - 1 < actions.size()) {
                int action = actions[i - 1];
                switch (action) {
                    case 0: break;
                    case 1: entities[i].vx += accel; break;
                    case 2: entities[i].vx -= accel; break;
                    case 3: entities[i].vy += accel; break;
                    case 4: entities[i].vy -= accel; break;
                    case 5: { 
                        Entity& ball = entities[0];
                        float dx = ball.x - entities[i].x;
                        float dy = ball.y - entities[i].y;
                        float dist = dx * dx + dy * dy;

                        if (dist < kick_range * kick_range) {
                            float kick_force = 10.0f;
                            float dist = std::sqrt(dist);
                            ball.vx += (dx / dist) * kick_force;
                            ball.vy += (dy / dist) * kick_force;
                        }
                        break;
                    }
                }
                entities[i].vx = std::clamp(entities[i].vx, -top_speed, top_speed);
                entities[i].vy = std::clamp(entities[i].vy, -top_speed, top_speed);
            }
        }
    }

    void update_positions() {
        for (auto& entity : entities) {
            entity.x += entity.vx * dt;
            entity.y += entity.vy * dt;

            entity.vx *= 0.95f;
            entity.vy *= 0.95f;
            
            handle_boundaries(entity);
        }
    }

    void handle_boundaries(Entity& entity) {
        bool in_goal_height = (entity.y > height * 0.3f && entity.y < height * 0.7f);
        
        if (entity.x - entity.radius < 0) {
            entity.x = entity.radius;
            if (entity.type == BALL && !in_goal_height) entity.vx = -entity.vx * 0.8f;
            else entity.vx = 0;
        }
        if (entity.x + entity.radius > width) {
            entity.x = width - entity.radius;
            if (entity.type == BALL && !in_goal_height) entity.vx = -entity.vx * 0.8f;
            else entity.vx = 0;
        }
        if (entity.y - entity.radius < 0) {
            entity.y = entity.radius;
            if (entity.type != BALL) entity.vy = 0;
            else entity.vy = -entity.vy * 0.8f;
        }
        if (entity.y + entity.radius > height) {
            entity.y = height - entity.radius;
            if (entity.type != BALL) entity.vy = 0;
            else entity.vy = -entity.vy * 0.8f;
        }
    }

    void handle_collisions() {
        for (size_t i = 0; i < entities.size(); ++i) {
            for (size_t j = i + 1; j < entities.size(); ++j) {
                resolve_collision(entities[i], entities[j]);
            }
        }
    }

    void resolve_collision(Entity& a, Entity& b) {
        float dx = b.x - a.x;
        float dy = b.y - a.y;
        float dist_sq = dx * dx + dy * dy;
        float radius_sum = a.radius + b.radius;

        if (dist_sq < radius_sum * radius_sum) {
            float dist = std::sqrt(dist_sq);
            if (dist == 0) return;

            float overlap = 0.5f * (radius_sum - dist);
            float nx = dx / dist;
            float ny = dy / dist;

            a.x -= overlap * nx;
            a.y -= overlap * ny;
            b.x += overlap * nx;
            b.y += overlap * ny;

            float rvx = b.vx - a.vx;
            float rvy = b.vy - a.vy;
            float vel_along_normal = rvx * nx + rvy * ny;

            if (vel_along_normal > 0) return;

            float restitution = (a.type == BALL || b.type == BALL) ? 0.9f : 0.2f;
            float j_impulse = -(1.0f + restitution) * vel_along_normal;
            j_impulse /= (1.0f / a.mass + 1.0f / b.mass);

            a.vx -= (1.0f / a.mass) * j_impulse * nx;
            a.vy -= (1.0f / a.mass) * j_impulse * ny;
            b.vx += (1.0f / b.mass) * j_impulse * nx;
            b.vy += (1.0f / b.mass) * j_impulse * ny;
        }
    }

    void check_goals() {
        Entity& ball = entities[0];
        
        if (ball.x < 0 && ball.y > height * 0.3f && ball.y < height * 0.7f) {
            steps = max_steps;
        }
        if (ball.x > width && ball.y > height * 0.3f && ball.y < height * 0.7f) {
            steps = max_steps;
        }
    }

    void update_distance_tracking() {
        Entity& ball = entities[0];
        prev_ball_dist_to_goal_b = std::abs(ball.x - width);
        prev_ball_dist_to_goal_a = std::abs(ball.x);
        
        prev_player_ball_distances.clear();
        for (size_t i = 1; i < entities.size(); ++i) {
            float dx = entities[i].x - ball.x;
            float dy = entities[i].y - ball.y;
            prev_player_ball_distances.push_back(std::sqrt(dx * dx + dy * dy));
        }
    }

    std::vector<float> get_rewards() {
        std::vector<float> rewards(2, 0.0f);
        Entity& ball = entities[0];
        
        const float goal_bonus = 10.0f;
        const float step_penalty = -0.1f;
        const float boundary_penalty = -0.5f;
        const float ball_approach_reward = 0.1f;
        const float touch_reward = 0.1f;

        // Check for goals and assign rewards
        if (ball.x < 0 && ball.y > height * 0.3f && ball.y < height * 0.7f) {
            rewards[1] += goal_bonus;
        }
        if (ball.x > width && ball.y > height * 0.3f && ball.y < height * 0.7f) {
            rewards[0] += goal_bonus;
        }

        // Base step penalty
        rewards[0] += step_penalty;
        rewards[1] += step_penalty;

        // Ball approach to opponent goal reward
        float curr_dist_to_goal_b = std::abs(ball.x - width);
        float curr_dist_to_goal_a = std::abs(ball.x);
        
        if (curr_dist_to_goal_b < prev_ball_dist_to_goal_b) {
            rewards[0] += ball_approach_reward;
        }
        if (curr_dist_to_goal_a < prev_ball_dist_to_goal_a) {
            rewards[1] += ball_approach_reward;
        }
        
        prev_ball_dist_to_goal_b = curr_dist_to_goal_b;
        prev_ball_dist_to_goal_a = curr_dist_to_goal_a;

        // Player-ball distance and touch rewards
        for (size_t i = 1; i < entities.size(); ++i) {
            float dx = entities[i].x - ball.x;
            float dy = entities[i].y - ball.y;
            float curr_dist = std::sqrt(dx * dx + dy * dy);
            size_t player_idx = i - 1;
            
            if (player_idx < prev_player_ball_distances.size()) {
                // Reward for getting closer to ball
                if (curr_dist < prev_player_ball_distances[player_idx]) {
                    if (entities[i].type == PLAYER_A) {
                        rewards[0] += ball_approach_reward * 0.5f;
                    } else {
                        rewards[1] += ball_approach_reward * 0.5f;
                    }
                }
                
                // Bonus for touching the ball
                if (curr_dist < entities[i].radius + ball.radius) {
                    if (entities[i].type == PLAYER_A) {
                        rewards[0] += touch_reward;
                    } else {
                        rewards[1] += touch_reward;
                    }
                }
                
                prev_player_ball_distances[player_idx] = curr_dist;
            }
        }

        // Boundary hit penalty for players
        for (size_t i = 1; i < entities.size(); ++i) {
            if ((entities[i].x - entities[i].radius <= 0 || entities[i].x + entities[i].radius >= width ||
                 entities[i].y - entities[i].radius <= 0 || entities[i].y + entities[i].radius >= height)) {
                if (entities[i].type == PLAYER_A) {
                    rewards[0] += boundary_penalty;
                } else {
                    rewards[1] += boundary_penalty;
                }
            }
        }

        return rewards;
    }
};

PYBIND11_MODULE(soccer_sim, m) {
    py::class_<SoccerEnv>(m, "SoccerEnv")
        .def(py::init<int, int, float, float, float>())
        .def("step", &SoccerEnv::step)
        .def("reset", &SoccerEnv::reset)
        .def("get_state", &SoccerEnv::get_state)
        .def("is_done", &SoccerEnv::is_done);
}
