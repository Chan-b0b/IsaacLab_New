#pragma once

#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick.hpp"

/**
 * @brief Bridge that maps keyboard keys to joystick button states.
 * This allows keyboard input to work with the FSM's joystick-based DSL parser.
 */
class KeyboardJoystickBridge {
public:
    KeyboardJoystickBridge(std::shared_ptr<Keyboard> kb) : keyboard_(kb) {}
    
    void update(unitree::common::UnitreeJoystick& joy) {
        if (!keyboard_) return;
        
        keyboard_->update();
        std::string key = keyboard_->key();
        
        // Map keyboard keys to joystick buttons
        // A key -> A button
        if (key == "a" && keyboard_->on_pressed) {
            joy.A.set(1);
        } else if (key != "a") {
            joy.A.set(0);
        }
        
        // B key -> B button  
        if (key == "b" && keyboard_->on_pressed) {
            joy.B.set(1);
        } else if (key != "b") {
            joy.B.set(0);
        }
        
        // X key -> X button
        if (key == "x" && keyboard_->on_pressed) {
            joy.X.set(1);
        } else if (key != "x") {
            joy.X.set(0);
        }
    }

private:
    std::shared_ptr<Keyboard> keyboard_;
};
