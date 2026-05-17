#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

// Run this cmd to build: c++ -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup $(python3 -m pybind11 --includes) src/envs/soccer_envs/soccer_env.cpp -o src/envs/soccer_envs/soccer_sim.so

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
    float width, height, dt, goal_size;
    int num_team_a, num_team_b;
    int steps = 0;
    
    // Configurable parameters
    float kick_range;
    float top_speed;
    float accel;
    float kick_force;
    float ball_restitution;
    float player_restitution;
    float friction;
    float goal_bonus;
    float step_penalty;
    float ball_move_weight;
    float player_move_weight;
    float contact_reward;
    float out_of_bounds_penalty;
    float kick_reward_weight;
    int max_steps;
    bool random_ball_placement;

    std::mt19937 rng;

    // Track previous state for reward calculation
    float dist_ball_to_goal_a = 0;
    float dist_ball_to_goal_b = 0;
    float dist_to_ball = 0;

    float prev_ball_dist_to_goal_b = 0;
    float prev_ball_dist_to_goal_a = 0;
    std::vector<float> prev_player_ball_distances;
    std::vector<float> last_kick_rewards = {0.0f, 0.0f};

    SoccerEnv(int num_a, int num_b, float w = 80.0, float h = 40.0, float time_step = 0.1, float g_size = 20.0f,
              float kr = 2.0f, float ts = 5.0f, float ac = 0.5f, float kf = 20.0f,
              float br = 0.9f, float pr = 0.2f, float fric = 0.8f,
              float gb = 10.0f, float sp = -0.1f, float bmw = 0.5f,
              float pmw = 0.2f, float cr = 0.1f, float obp = -1.0f, 
              float krw = 2.0f, int max_steps = 1000, bool random_ball_spawn = false)
        : width(w), height(h), dt(time_step), goal_size(g_size), 
          num_team_a(num_a), num_team_b(num_b),
          kick_range(kr), top_speed(ts), accel(ac), kick_force(kf),
          ball_restitution(br), player_restitution(pr), friction(fric),
          goal_bonus(gb), step_penalty(sp), ball_move_weight(bmw),
          player_move_weight(pmw), contact_reward(cr), out_of_bounds_penalty(obp), 
          kick_reward_weight(krw), max_steps(max_steps), random_ball_placement(random_ball_spawn),
          rng(std::random_device{}()) {
                reset();
            }

    void reset() {
        entities.clear();
        steps = 0;
        
        float ball_x = width / 2.0f;
        float ball_y = height / 2.0f;
        if (random_ball_placement) {
            // Spawn in middle 50% of field dimensions.
            std::uniform_real_distribution<float> x_dist(width * 0.25f, width * 0.75f);
            std::uniform_real_distribution<float> y_dist(height * 0.25f, height * 0.75f);
            ball_x = x_dist(rng);
            ball_y = y_dist(rng);
        }
        entities.push_back({ball_x, ball_y, 0, 0, 0.5f, 0.05f, BALL, 0});

        for (int i = 0; i < num_team_a; ++i) {
            float y = height / (num_team_a + 1) * (i + 1);
            entities.push_back({width / 4, y, 0, 0, 1.0f, 1.0f, PLAYER_A, i});
        }

        for (int i = 0; i < num_team_b; ++i) {
            float y = height / (num_team_b + 1) * (i + 1);
            entities.push_back({3 * width / 4, y, 0, 0, 1.0f, 1.0f, PLAYER_B, i});
        }
        
        reset_distance_tracking();
    }

    bool is_in_goal_range(float y) {
        float goal_top = (height + goal_size) / 2.0f;
        float goal_bottom = (height - goal_size) / 2.0f;
        return (y >= goal_bottom && y <= goal_top);
    }

    void step(std::vector<int> actions) {
        update_prev_state();
        apply_actions(actions);
        update_positions();
        handle_collisions();
        check_goals();
        
        steps++;
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

    std::vector<float> get_rewards() {
        std::vector<float> rewards(2, 0.0f);
        Entity& ball = entities[0];
        
        float goal_a_x = 0, goal_a_y = height / 2.0f;
        float goal_b_x = width, goal_b_y = height / 2.0f;

        dist_ball_to_goal_a = std::hypot(ball.x - goal_a_x, ball.y - goal_a_y);
        dist_ball_to_goal_b = std::hypot(ball.x - goal_b_x, ball.y - goal_b_y);
        
        // Goal Reward
        if (ball.x <= 0 && is_in_goal_range(ball.y)) rewards[1] += goal_bonus;
        if (ball.x >= width && is_in_goal_range(ball.y)) rewards[0] += goal_bonus;

        // Ball approach to opponent goal reward
        rewards[0] += (prev_ball_dist_to_goal_b - dist_ball_to_goal_b) * ball_move_weight;
        rewards[1] += (prev_ball_dist_to_goal_a - dist_ball_to_goal_a) * ball_move_weight;


        // Kick rewards
        rewards[0] += last_kick_rewards[0];
        rewards[1] += last_kick_rewards[1];

        // Player-ball distance and touch rewards
        for (size_t i = 1; i < entities.size(); ++i) {
            Entity& p = entities[i];
            int team_idx = (p.type == PLAYER_A) ? 0 : 1;
            
            dist_to_ball = std::hypot(p.x - ball.x, p.y - ball.y);
            size_t p_idx = i - 1;

            float diff = prev_player_ball_distances[p_idx] - dist_to_ball;
            rewards[team_idx] += diff * player_move_weight;
            // printf("team idx: %d\n", team_idx);
            // printf("Player %d distance to ball: %.2f, reward: %.3f\n", p.id, dist_to_ball, diff * player_move_weight);
            // printf("rewards: %.3f\n\n", rewards[team_idx]);

            if (dist_to_ball < p.radius + ball.radius + 0.1f) {
                float velocity_dir = ball.vx * (team_idx == 0 ? 1 : -1);
                if (velocity_dir > 0) {
                    rewards[team_idx] += contact_reward * (velocity_dir / top_speed);
                }
            }

            rewards[team_idx] += step_penalty; // Constant pressure to finish
            if (p.x <= p.radius + 0.1 || p.x >= width - p.radius - 0.1 || p.y <= p.radius + 0.1 || p.y >= height - p.radius - 0.1) {
                rewards[team_idx] += out_of_bounds_penalty;
            }
        }

        return rewards;
    }

    bool is_done() {
        Entity& ball = entities[0];
        bool scored = (ball.x <= 0 || ball.x >= width) && is_in_goal_range(ball.y);
        return steps >= max_steps || scored;
    }

private:
    void apply_actions(std::vector<int>& actions) {
        last_kick_rewards = {0.0f, 0.0f};
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
                            int team_idx = (entities[i].type == PLAYER_A) ? 0 : 1;
                            dist = std::sqrt(dist);
                            ball.vx += (dx / dist) * kick_force;
                            ball.vy += (dy / dist) * kick_force;

                            float target_goal_x = (team_idx == 0) ? width : 0;
                            float target_goal_y = height / 2.0f;
                            
                            float to_goal_x = target_goal_x - ball.x;
                            float to_goal_y = target_goal_y - ball.y;
                            float dist_to_goal = std::hypot(to_goal_x, to_goal_y);

                            // Dot product between kick direction and vector to goal
                            float dot = ((dx / dist) * (to_goal_x / dist_to_goal)) + 
                                        ((dy / dist) * (to_goal_y / dist_to_goal));

                            // Reward proportional to how well aimed the kick is (max 1.0)
                            last_kick_rewards[team_idx] += dot * kick_reward_weight; // Scale bonus as needed
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

            entity.vx *= friction;
            entity.vy *= friction;
            
            handle_boundaries(entity);
        }
    }

    void handle_boundaries(Entity& entity) {
        bool in_goal_y = is_in_goal_range(entity.y);
        
        if (entity.x - entity.radius < 0) {
            if (entity.type == BALL && in_goal_y) {
                // Let ball pass and trigger goal
            } else {
                entity.x = entity.radius;
                entity.vx *= -0.8f; 
            }
        }

        if (entity.x + entity.radius > width) {
            if (entity.type == BALL && in_goal_y) {
                // Let ball pass and trigger goal
            } else {
                entity.x = width - entity.radius;
                entity.vx *= -0.8f;
            }
        }

        if (entity.y - entity.radius < 0) {
            entity.y = entity.radius;
            entity.vy *= -0.8f;
        }
        if (entity.y + entity.radius > height) {
            entity.y = height - entity.radius;
            entity.vy *= -0.8f;
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
            if (dist < 1e-6f) return;

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

            float restitution = (a.type == BALL || b.type == BALL) ? ball_restitution : player_restitution;
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
        
        if ((ball.x <= 0 || ball.x >= width) && is_in_goal_range(ball.y)) {
            steps = max_steps;
        }
    }
    
    void update_prev_state() {
        Entity& ball = entities[0];
        prev_ball_dist_to_goal_a = std::hypot(ball.x - 0, ball.y - (height / 2.0f));
        prev_ball_dist_to_goal_b = std::hypot(ball.x - width, ball.y - (height / 2.0f));
        
        for (size_t i = 1; i < entities.size(); ++i) {
            prev_player_ball_distances[i-1] = std::hypot(entities[i].x - ball.x, entities[i].y - ball.y);
        }
    }

    void reset_distance_tracking() {
        Entity& ball = entities[0];
        // Goal A is at x=0, Goal B is at x=width
        prev_ball_dist_to_goal_a = std::hypot(ball.x - 0, ball.y - (height / 2.0f));
        prev_ball_dist_to_goal_b = std::hypot(ball.x - width, ball.y - (height / 2.0f));
        
        prev_player_ball_distances.assign(entities.size() - 1, 0.0f);
        for (size_t i = 1; i < entities.size(); ++i) {
            prev_player_ball_distances[i-1] = std::hypot(entities[i].x - ball.x, entities[i].y - ball.y);
        }
    }
};

PYBIND11_MODULE(soccer_sim, m) {
    py::class_<SoccerEnv>(m, "SoccerEnv")
        .def(py::init<int, int, float, float, float, float, float, float, float, float,
                      float, float, float, float, float, float, float, float, float, float, int, bool>(),
             py::arg("num_a"), 
             py::arg("num_b"), 
             py::arg("w") = 80.0f, 
             py::arg("h") = 40.0f, 
             py::arg("time_step") = 0.1f, 
             py::arg("g_size") = 20.0f,
             py::arg("kr") = 2.0f, 
             py::arg("ts") = 5.0f, 
             py::arg("ac") = 0.5f, 
             py::arg("kf") = 20.0f,
             py::arg("br") = 0.9f, 
             py::arg("pr") = 0.2f, 
             py::arg("fric") = 0.8f,
             py::arg("gb") = 10.0f, 
             py::arg("sp") = -0.1f, 
             py::arg("bmw") = 0.5f,
             py::arg("pmw") = 0.2f, 
             py::arg("cr") = 0.1f, 
            py::arg("obp") = -10.0f,
            py::arg("krw") = 2.0f,
            py::arg("max_steps") = 1000,
            py::arg("random_ball_spawn") = false)
        .def("step", &SoccerEnv::step)
        .def("reset", &SoccerEnv::reset)
        .def("get_state", &SoccerEnv::get_state)
        .def("is_done", &SoccerEnv::is_done)
        .def("get_rewards", &SoccerEnv::get_rewards);
}
