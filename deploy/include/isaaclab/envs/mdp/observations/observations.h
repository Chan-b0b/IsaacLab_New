// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data(27);
    
    // Use all_joint_pos which contains all 27 joints
    for(int i = 0; i < 27; i++) {
        float default_pos = (i < asset->data.default_joint_pos.size()) ? asset->data.default_joint_pos[i] : 0.0f;
        data[i] = asset->data.all_joint_pos[i] - default_pos;
    }
    
    static int print_counter = 0;
    if(print_counter++ % 50 == 0) {
        std::cout << "[joint_pos_rel]: ";
        for(int i = 0; i < std::min(12, (int)data.size()); i++) {
            std::cout << data[i];
            if(i < 11) std::cout << ", ";
        }
        std::cout << " ..." << std::endl;
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    
    static int print_counter = 0;
    if(print_counter++ % 50 == 0) {
        std::cout << "[joint_vel_rel]: ";
        for(int i = 0; i < std::min(12, (int)asset->data.all_joint_vel.size()); i++) {
            std::cout << asset->data.all_joint_vel[i];
            if(i < 11) std::cout << ", ";
        }
        std::cout << " ..." << std::endl;
    }
    
    // Return all_joint_vel which contains all 27 joints
    return asset->data.all_joint_vel;
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(6);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    obs[3] = 0.0f;  // walking_binary - always 0 (not walking)
    obs[4] = 0.8f; // height - default standing height in meters
    obs[5] = -0.2f; // waist_pitch - fixed value matching training

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

}
}