use sdl2::event::Event;

#[derive(Copy, Clone, Debug)]
pub struct Input {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    look_up: bool,
    look_down: bool,
    look_left: bool,
    look_right: bool,
    look_up_down: f32,
    look_left_right: f32,
    speed: f32,
    look_speed: f32,
}

impl Input {
    fn set_keycode(&mut self, keycode: &sdl2::keyboard::Keycode, v: bool) {
        match keycode {
            sdl2::keyboard::Keycode::W => self.forward = v,
            sdl2::keyboard::Keycode::S => self.backward = v,
            sdl2::keyboard::Keycode::A => self.left = v,
            sdl2::keyboard::Keycode::D => self.right = v,
            sdl2::keyboard::Keycode::J => self.look_left = v,
            sdl2::keyboard::Keycode::L => self.look_right = v,
            sdl2::keyboard::Keycode::K => self.look_down = v,
            sdl2::keyboard::Keycode::I => self.look_up = v,
            sdl2::keyboard::Keycode::Space => self.up = v,
            sdl2::keyboard::Keycode::LCtrl => self.down = v,
            sdl2::keyboard::Keycode::O => self.speed *= 1.1,
            sdl2::keyboard::Keycode::P => self.speed *= 0.9,
            _ => {}
        }
    }
    pub fn make_controls(&mut self, dt: f32) -> first_person_camera::Controls {
        let controls = first_person_camera::Controls {
            forwards_backwards: self.forward as i32 as f32 - self.backward as i32 as f32,
            right_left: self.right as i32 as f32 - self.left as i32 as f32,
            up_down: self.up as i32 as f32 - self.down as i32 as f32,
            look_up_down: -self.look_up_down * 0.005 + (self.look_up as i32 as f32 * self.look_speed * dt) - (self.look_down as i32 as f32 * self.look_speed * dt),
            look_right_left: -self.look_left_right * 0.005 + (self.look_left as i32 as f32 * self.look_speed * dt) - (self.look_right as i32 as f32 * self.look_speed * dt),
            speed: self.speed * dt,
        };
        self.look_up_down = 0.0;
        self.look_left_right = 0.0;
        controls
    }
    pub fn process_event(&mut self, event: &sdl2::event::Event) {
        match event {
            Event::MouseWheel { y, .. } => {
                self.speed *= 1.0 - (*y as f32 * -0.05);
            }
            Event::MouseMotion { xrel, yrel, .. } => {
                self.look_left_right += *xrel as f32;
                self.look_up_down += *yrel as f32;
            }
            Event::KeyUp {
                keycode: Some(keycode),
                ..
            } => self.set_keycode(keycode, false),
            Event::KeyDown {
                keycode: Some(keycode),
                ..
            } => self.set_keycode(keycode, true),
            _ => {}
        }
    }
}

impl Default for Input {
    fn default() -> Self {
        Self {
            speed: 1.0,
            forward: Default::default(),
            backward: Default::default(),
            left: Default::default(),
            right: Default::default(),
            up: Default::default(),
            down: Default::default(),
            look_up_down: Default::default(),
            look_left_right: Default::default(),
            look_up: false,
            look_down: false,
            look_right: false,
            look_left: false,
            look_speed: 2.0
        }
    }
}
