// #![allow(unused)]
// https://uwe-repository.worktribe.com/output/980579
// https://cargocollective.com/sagejenson/physarum
//
// `cannot borrow `*self` as mutable because it is also borrowed as immutable` =>
// http://smallcultfollowing.com/babysteps/blog/2018/11/01/after-nll-interprocedural-conflicts/

extern crate image;

use std::collections::HashSet;
use std::io::{stdout, Write};
use std::time::Duration;

use crossterm::event::{Event, KeyCode, KeyEvent};
use crossterm::style;
use crossterm::terminal::ClearType;
use crossterm::{cursor, event, queue, terminal};
use image::{GrayImage, Luma, Rgb, RgbImage};
#[allow(unused_imports)]
use palette::{Gradient, IntoColor, LinSrgb, LinSrgba, Mix};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

const OUTPUT: bool = true;
const OUTPUT_FRAMES: bool = false;
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const CLEAR_SCREEN: bool = true;
const PREVENT_OVERLAP: bool = false;
const ITERATIONS: u32 = 300;
const ITERATION_DELAY: Duration = Duration::from_millis(5);

const POPULATION_PCT: u32 = 10; // 5
const AGENT_COUNT: u32 = (WIDTH as f32 * HEIGHT as f32 * (POPULATION_PCT as f32 / 100.0)) as u32;
const DECAY_FACTOR: f32 = 0.1;

const SENSOR_ANGLE: f32 = 45.0 / 2.0; // degrees
const SENSOR_WIDTH: u32 = 5; // pixels (must be odd)
const SENSOR_OFFSET: u32 = 9; // pixels
const ROTATION_ANGLE: f32 = 22.5 / 3.0; // degrees
const DIFFUSION_KERNEL_SIZE: u8 = 3; // pixels (must be odd)
const STEP_SIZE: u32 = 1; // pixels
const DEPOSITION: f32 = 5.0;
const DEPOSITION_RAMP: u32 = 0;
const RANDOM_DIRECTION_P: u32 = 0; // 0-100
const TRAIL_SENSITIVITY: f32 = 0.0;
const INITIAL_FOOD: f32 = 0.0;

const DEBUG_AGENTS: bool = false;
const DEBUG_TRAILS: bool = true;
const DEBUG_TRAIL_VALUES: bool = false;

type TrailMap = Vec<Vec<f32>>;
type Point = (usize, usize);

// nullify sensors that probe disabled cells
type Mask = Vec<Vec<bool>>;

#[allow(unused)]
#[derive(Debug)]
struct Population {
    label: String,
    agents: Vec<Agent>,
    occupied_positions: HashSet<Point>,
    kernels: Vec<Vec<(usize, usize)>>,
    trails: TrailMap,
    mask: Mask,
}

#[derive(Debug)]
struct Agent {
    pos: (f32, f32),
    angle: f32,
}

#[derive(Debug)]
struct Simulation {
    occupied_positions: HashSet<Point>,
    iteration: u32,
}

impl Simulation {
    fn new() -> Self {
        Simulation {
            occupied_positions: Default::default(),
            iteration: 0,
        }
    }

    fn step(&mut self, populations: &mut [Population]) -> crossterm::Result<()> {
        if CLEAR_SCREEN {
            clear_screen()?;
        }

        self.iteration += 1;

        for population in populations.iter_mut() {
            population.step(self, self.iteration);

            if DEBUG_AGENTS && !OUTPUT {
                debug_agents(&population.agents)?;
            }

            if DEBUG_TRAILS && !OUTPUT {
                debug_trails(&population.trails)?;
            }
        }

        queue!(
            stdout(),
            style::Print(format!("Iteration: {}", self.iteration)),
            cursor::MoveToNextLine(1)
        )?;

        flush()?;

        if !OUTPUT {
            std::thread::sleep(ITERATION_DELAY);
        }

        Ok(())
    }
}

impl Population {
    fn new(label: String, mask: Vec<Vec<bool>>) -> Self {
        Population {
            label,
            agents: Default::default(),
            occupied_positions: Default::default(),
            kernels: get_diffusion_kernels(),
            trails: vec![vec![INITIAL_FOOD; WIDTH as usize]; HEIGHT as usize],
            mask,
        }
    }

    fn add_agents(&mut self, agents: Vec<Agent>) {
        self.agents.extend(agents);
    }

    fn step(&mut self, sim: &mut Simulation, iteration: u32) {
        self.agents.shuffle(&mut thread_rng());

        //
        // 1. simulate each agent
        //
        for agent in self.agents.iter_mut() {
            agent.step(
                sim,
                iteration,
                &mut self.occupied_positions,
                &mut self.trails,
                &self.mask,
            );
        }

        let snapshot = self.trails.clone();

        //
        // 2. diffuse
        //  - visit every pixel in trail map and disperse its value to its neighbors
        //    via a mean filter average
        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                let kernel = &self.kernels[(y * WIDTH + x) as usize];
                self.trails[y as usize][x as usize] = mean_filter(&snapshot, kernel);
            }
        }

        //
        // 3. decay
        //
        decay(&mut self.trails);
    }
}

impl Agent {
    fn new(pos: (f32, f32), angle: f32) -> Agent {
        Agent { pos, angle }
    }

    fn step(
        &mut self,
        sim: &mut Simulation,
        iteration: u32,
        occupied_positions: &mut HashSet<Point>,
        trails: &mut TrailMap,
        mask: &Mask,
    ) {
        let rng = &mut thread_rng();

        // 1. try to move
        if let Err(_) = self.try_move(sim, occupied_positions) {
            self.angle = random_rotation(self.angle, rng);
            return;
        }

        // 2. sense + rotate
        self.rotate(trails, mask);

        // 3. deposit
        self.deposit(trails, mask, iteration);
    }

    fn try_move(
        &mut self,
        sim: &mut Simulation,
        occupied_positions: &mut HashSet<Point>,
    ) -> Result<(), ()> {
        let next_point = translate(self.pos, self.angle, STEP_SIZE);

        // maybe failed to move
        if PREVENT_OVERLAP && is_occupied(quantize(wrap(next_point)), sim, &occupied_positions) {
            return Err(());
        }

        // moved successfully
        depart(quantize(self.pos), sim, occupied_positions);
        self.pos = wrap(next_point);
        occupy(quantize(self.pos), sim, occupied_positions);

        Ok(())
    }

    fn probe(&self, trails: &TrailMap, mask: &Mask) -> (f32, f32, f32) {
        let sl = wrap(translate(
            self.pos,
            normalize_angle(self.angle - SENSOR_ANGLE),
            SENSOR_OFFSET,
        ));
        let sm = wrap(translate(self.pos, self.angle, SENSOR_OFFSET));
        let sr = wrap(translate(
            self.pos,
            normalize_angle(self.angle + SENSOR_ANGLE),
            SENSOR_OFFSET,
        ));

        let l = if !mask[sl.1 as usize][sl.0 as usize] {
            0.0
        } else {
            sense(sl, trails)
        };

        let m = if !mask[sm.1 as usize][sm.0 as usize] {
            0.0
        } else {
            sense(sm, trails)
        };

        let r = if !mask[sr.1 as usize][sr.0 as usize] {
            0.0
        } else {
            sense(sr, trails)
        };

        (l, m, r)
    }

    fn rotate(&mut self, trails: &TrailMap, mask: &Mask) {
        let (l, m, r) = self.probe(trails, mask);
        let rng = &mut thread_rng();

        if m > l && m > r {
            // center is strongest -> no turn
        } else if m < l && m < r {
            // center is weakest -> random angle
            self.angle = random_rotation(self.angle, rng);
        } else if l < r {
            // right is strongest -> turn right
            self.angle = normalize_angle(self.angle + ROTATION_ANGLE);
        } else {
            // left is strongest -> turn left
            self.angle = normalize_angle(self.angle - ROTATION_ANGLE);
        }

        if RANDOM_DIRECTION_P > 0 {
            self.angle = normalize_angle(self.angle + maybe_turn(rng));
        }
    }

    fn deposit(&mut self, trails: &mut TrailMap, mask: &Mask, iteration: u32) {
        let (x, y) = quantize(self.pos);
        let scale = (iteration as f32 / DEPOSITION_RAMP as f32).min(1.0);

        if mask[y as usize][x as usize] {
            trails[y as usize][x as usize] += DEPOSITION * scale;
        } else {
            trails[y as usize][x as usize] = 0.0;
        }
    }
}

fn is_occupied(pos: Point, sim: &Simulation, occupied_positions: &HashSet<Point>) -> bool {
    occupied_positions.contains(&pos) || sim.occupied_positions.contains(&pos)
}

fn occupy(pos: Point, sim: &mut Simulation, occupied_positions: &mut HashSet<Point>) {
    occupied_positions.insert(pos);
    sim.occupied_positions.insert(pos);
}

fn depart(pos: Point, sim: &mut Simulation, occupied_positions: &mut HashSet<Point>) {
    occupied_positions.remove(&pos);
    sim.occupied_positions.remove(&pos);
}

struct LeArtiste {
    palettes: Vec<Gradient<LinSrgba>>,
}

impl LeArtiste {
    fn new() -> Self {
        LeArtiste {
            palettes: Default::default(),
        }
    }

    fn add_palette(&mut self, palette: Gradient<LinSrgba>) {
        self.palettes.push(palette);
    }

    fn draw(&self, populations: &[Population], path: String) {
        let mut image = RgbImage::new(WIDTH, HEIGHT);
        let pixels = self.get_pixels(populations);

        self.draw_pixels(&mut image, &pixels);

        image.save(path).unwrap();
    }

    fn get_pixels(&self, populations: &[Population]) -> Vec<Vec<Option<LinSrgb>>> {
        let mut pixels: Vec<Vec<Option<LinSrgb>>> =
            vec![vec![None; WIDTH as usize]; HEIGHT as usize];

        for (i, population) in populations.iter().enumerate() {
            let mut max: f32 = 0.0;

            // normalize trails
            for x in 0..WIDTH {
                for y in 0..HEIGHT {
                    for population in populations {
                        max = max.max(population.trails[y as usize][x as usize]);
                    }
                }
            }

            for x in 0..WIDTH {
                for y in 0..HEIGHT {
                    let v = population.trails[y as usize][x as usize];
                    let p = v / max; // normalized value

                    if p < 0.005 {
                        continue;
                    }

                    let color: LinSrgb = self.palettes[i].get(p).into_color();

                    // blend colors
                    // let result = if let Some(current) = pixels[y as usize][x as usize] {
                    //     Mix::mix(&current, &other, 0.5)
                    // } else {
                    //     other
                    // };
                    // pixels[y as usize][x as usize] = Some(result);

                    pixels[y as usize][x as usize] = Some(color);
                }
            }
        }

        // save output
        pixels
    }

    fn draw_pixels(&self, image: &mut RgbImage, pixels: &[Vec<Option<LinSrgb>>]) {
        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                if let Some(pixel) = pixels[y as usize][x as usize] {
                    let (r, g, b) = pixel.into_components();

                    image.put_pixel(
                        x,
                        y,
                        Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]),
                    )
                }
            }
        }
    }
}

fn clear_screen() -> crossterm::Result<()> {
    queue!(
        stdout(),
        cursor::Hide,
        terminal::Clear(ClearType::All),
        cursor::MoveTo(0, 0)
    )?;
    Ok(())
}

fn flush() -> crossterm::Result<()> {
    queue!(stdout(), cursor::MoveTo(0, 0))?;
    stdout().flush()
}

fn cleanup() -> crossterm::Result<()> {
    queue!(stdout(), cursor::Show)?;
    stdout().flush()?;
    terminal::disable_raw_mode()
}

fn is_event_available() -> crossterm::Result<bool> {
    event::poll(Duration::from_secs(0))
}

fn read_char() -> crossterm::Result<Option<char>> {
    if is_event_available()? {
        if let Ok(Event::Key(KeyEvent {
            code: KeyCode::Char(c),
            ..
        })) = event::read()
        {
            return Ok(Some(c));
        }
    }

    Ok(None)
}

fn main() -> crossterm::Result<()> {
    terminal::enable_raw_mode()?;

    let mut artist = LeArtiste::new();

    for palette in get_palettes() {
        artist.add_palette(palette);
    }

    // let mut mask1_image = GrayImage::new(WIDTH, HEIGHT);
    // let mut mask2_image = GrayImage::new(WIDTH, HEIGHT);

    let mut mask1 = vec![vec![true; WIDTH as usize]; HEIGHT as usize];
    let mut mask2 = vec![vec![true; WIDTH as usize]; HEIGHT as usize];

    let x0 = (WIDTH as f32 / 2.5) as u32;
    let y0 = (HEIGHT as f32 / 1.8) as u32;
    let size = WIDTH.min(HEIGHT) / 4;

    draw_circle_mask(&mut mask1, x0, y0, size as i32);
    draw_circle_mask(
        &mut mask2,
        x0 + (size as f32 * 1.4) as u32,
        y0 - size,
        (size / 2) as i32,
    );

    // draw_mask_image(&mut mask1_image, &mask1);
    // draw_mask_image(&mut mask2_image, &mask2);

    // mask1_image.save("big.png").unwrap();
    // mask2_image.save("small.png").unwrap();

    let mut sim = Simulation::new();
    let mut populations = vec![];

    let mut ricks = Population::new("Ricks".to_string(), mask1);
    let mut morties = Population::new("Morties".to_string(), mask2);

    add_random_agents(&mut ricks, AGENT_COUNT);
    add_random_agents(&mut morties, AGENT_COUNT);

    populations.push(ricks);
    populations.push(morties);

    // simulate
    for i in 0..ITERATIONS {
        sim.step(&mut populations)?;

        if OUTPUT_FRAMES {
            artist.draw(&populations, format!("frames/{:05}.png", i));
        }

        match read_char()? {
            Some('q') => break,
            _ => continue,
        };
    }

    if OUTPUT {
        artist.draw(&populations, "art.png".to_string());
    }

    cleanup()?;
    Ok(())
}

// viget blue: 69, 147, 182 | #4594B6
// viget orange: 225, 116, 56 | #e17338
fn get_palettes() -> Vec<Gradient<LinSrgba>> {
    let mut palettes = vec![];

    {
        // let start = LinSrgba::new(0.0, 0.0, 0.0, 0.0);
        // let mid = LinSrgba::new(69.0 / 255.0, 147.0 / 255.0, 182.0 / 255.0, 1.0);
        // let end = LinSrgba::new(255.0, 255.0, 255.0, 1.0);
        let low = LinSrgba::new(0.0, 0.0, 0.0, 0.0);
        let mid = LinSrgba::new(69.0 / 255.0, 147.0 / 255.0, 182.0 / 255.0, 1.0);
        let high = LinSrgba::new(221.0 / 255.0, 234.0 / 255.0, 215.0 / 255.0, 1.0);

        let gradient: Gradient<LinSrgba> =
            // Gradient::with_domain(vec![(0.0, start), (0.7, mid), (1.0, end)]);
            // Gradient::with_domain(vec![(0.0, start), (0.8, mid)]);
            Gradient::with_domain(vec![(0.0, low), (0.3, mid), (0.9, high)]);

        palettes.push(gradient);
    }

    {
        // let start = LinSrgba::new(0.0, 0.0, 0.0, 0.0);
        // // let mid = LinSrgba::new(69.0 / 255.0, 147.0 / 255.0, 182.0 / 255.0, 0.5);
        // let mid = LinSrgba::new(10.0 / 255.0, 67.0 / 255.0, 91.0 / 255.0, 1.0);
        // let end = LinSrgba::new(225.0 / 255.0, 116.0 / 255.0, 56.0 / 255.0, 1.0);
        // let high = LinSrgba::new(228.0 / 255.0, 165.0 / 255.0, 131.0 / 255.0, 1.0);

        let low = LinSrgba::new(0.0, 0.0, 0.0, 0.0);
        let mid = LinSrgba::new(91.0 / 255.0, 35.0 / 255.0, 51.0 / 255.0, 1.0);
        let high = LinSrgba::new(239.0 / 255.0, 233.0 / 255.0, 174.0 / 255.0, 1.0);

        // let end = LinSrgba::new(255.0, 255.0, 255.0, 1.0);
        let gradient: Gradient<LinSrgba> =
            // Gradient::with_domain(vec![(0.0, start), (0.7, mid), (1.0, end)]);
            // Gradient::with_domain(vec![(0.0, start), (0.1, mid), (0.5, end), (1.0, high)]);
            Gradient::with_domain(vec![(0.0, low), (0.3, mid), (0.9, high)]);

        palettes.push(gradient);
    }

    palettes
}

fn add_random_agents(population: &mut Population, count: u32) {
    let rng = &mut thread_rng();
    let mut agents: Vec<Agent> = vec![];

    for _ in 0..count {
        let x = rng.gen_range(0..WIDTH) as f32;
        let y = rng.gen_range(0..HEIGHT) as f32;
        let angle = rng.gen_range(0..360) as f32;

        agents.push(Agent::new((x, y), angle));
    }

    population.add_agents(agents);
}

fn mean_filter(snapshot: &TrailMap, kernel: &[Point]) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;

    for (x, y) in kernel {
        let x = *x as usize;
        let y = *y as usize;

        if x < WIDTH as usize && y < HEIGHT as usize {
            sum += snapshot[y][x];
            count += 1;
        }
    }

    sum / count as f32
}

fn decay(trail_map: &mut TrailMap) {
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            trail_map[y as usize][x as usize] *= 1.0 - DECAY_FACTOR;
        }
    }
}

fn get_diffusion_kernels() -> Vec<Vec<Point>> {
    let mut kernels = vec![];
    let o = ((DIFFUSION_KERNEL_SIZE - 1) / 2) as i32;

    for y in 0..HEIGHT as i32 {
        for x in 0..WIDTH as i32 {
            let mut kernel = vec![];

            for xx in (x - o)..=(x + o) {
                for yy in (y - o)..=(y + o) {
                    kernel.push(quantize(wrap((xx as f32, yy as f32))));
                }
            }

            kernels.push(kernel);
        }
    }

    kernels
}

fn radians(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

fn translate(origin: (f32, f32), degrees: f32, d: u32) -> (f32, f32) {
    let (mut x, mut y) = origin;
    let angle = radians(degrees);

    x = x + (angle.cos() * d as f32);
    y = y + (angle.sin() * d as f32);

    (x, y)
}

fn wrap(position: (f32, f32)) -> (f32, f32) {
    let (mut x, mut y) = position;
    let h = HEIGHT as f32;
    let w = WIDTH as f32;

    if x.round() < 0.0 {
        x += w;
    } else if x.round() >= w {
        x -= w;
    }

    if y.round() < 0.0 {
        y += h;
    } else if y.round() >= h {
        y -= h;
    }

    (x, y)
}

fn random_rotation(degrees: f32, rng: &mut impl Rng) -> f32 {
    let next_angle: f32;

    if rng.gen_ratio(1, 2) {
        next_angle = (degrees - ROTATION_ANGLE) % 360.0;
    } else {
        next_angle = (degrees + ROTATION_ANGLE) % 360.0;
    }

    normalize_angle(next_angle)
}

fn maybe_turn(rng: &mut impl Rng) -> f32 {
    if rng.gen_ratio(RANDOM_DIRECTION_P, 100) {
        if rng.gen_ratio(1, 2) {
            ROTATION_ANGLE
        } else {
            -ROTATION_ANGLE
        }
    } else {
        0.0
    }
}

fn normalize_angle(degrees: f32) -> f32 {
    if degrees <= 0.0 {
        degrees + 360.0
    } else if degrees > 360.0 {
        degrees - 360.0
    } else {
        degrees
    }
}

fn sense(position: (f32, f32), trail_map: &TrailMap) -> f32 {
    let (x, y) = position;
    let o = ((SENSOR_WIDTH - 1) / 2) as i32;
    let mut total = 0.0;

    if o == 0 {
        total += trail_map[y as usize][x as usize];
    } else {
        for xx in (x as i32 - o)..=(x as i32 + o) {
            for yy in (y as i32 - o)..=(y as i32 + o) {
                let (xxx, yyy) = quantize(wrap((xx as f32, yy as f32)));
                total += trail_map[yyy][xxx];
            }
        }
    }

    if total >= TRAIL_SENSITIVITY {
        total
    } else {
        0.0
    }
}

fn quantize(p: (f32, f32)) -> Point {
    (p.0 as usize, p.1 as usize)
}

fn debug_agents(agents: &[Agent]) -> crossterm::Result<()> {
    // TEMP: visualize agent positions for now.. just making sure things
    // are doing what I expect so far
    let mut frame = [[0u8; WIDTH as usize]; HEIGHT as usize];

    for agent in agents {
        let (x, y) = agent.pos;
        frame[y as usize][x as usize] = 1;
    }

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if frame[y as usize][x as usize] == 1 {
                queue!(stdout(), style::Print(" o"))?;
            } else {
                queue!(stdout(), style::Print("  "))?;
            }
        }

        queue!(stdout(), cursor::MoveToNextLine(1))?;
    }

    Ok(())
}

fn debug_trails(trail_map: &TrailMap) -> crossterm::Result<()> {
    let tex = [" ", "░", "▒", "▓", "█"];
    let mut max = 0.0;

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            max = trail_map[y as usize][x as usize].max(max);
        }
    }

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let v = trail_map[y as usize][x as usize];

            if DEBUG_TRAIL_VALUES {
                if v > 0.0 {
                    queue!(stdout(), style::Print(format!(" {:^5.1} ", v)))?;
                } else {
                    queue!(stdout(), style::Print("   .   "))?;
                }
            } else {
                let idx = (v * 4.0 / max) as usize;
                queue!(stdout(), style::Print(format!(" {}", tex[idx])))?;
            }
        }

        queue!(stdout(), cursor::MoveToNextLine(1))?;
    }

    Ok(())
}

#[allow(unused)]
fn draw_mask_image(image: &mut GrayImage, mask: &Mask) {
    for (y, row) in mask.iter().enumerate() {
        for (x, v) in row.iter().enumerate() {
            if *v {
                image.put_pixel(x as u32, y as u32, Luma([255u8]));
            }
        }
    }
}

fn draw_circle_mask(mask: &mut Mask, x0: u32, y0: u32, r: i32) {
    let r2 = r * r;

    let x_pos = |o: i32| (x0 as i32 + o) as u32;
    let y_pos = |o: i32| (y0 as i32 + o) as u32;

    for y in -r..r {
        for x in -r..r {
            if x * x + y * y <= r2 {
                mask[y_pos(y) as usize][x_pos(x) as usize] = true;
            }
        }
    }
}
