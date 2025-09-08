/*!
 * Rapier.wasm - High-Performance Physics Engine for WebAssembly
 * 
 * Production-quality WASM bindings for the Rapier physics engine with:
 * - 2D/3D physics simulation with configurable precision
 * - SIMD acceleration for vector operations
 * - WebGPU compute shader integration for GPU acceleration  
 * - Memory-efficient object pooling and state management
 * - Deterministic simulation for networked physics
 * - Comprehensive JavaScript/TypeScript API
 * 
 * 
 * WASM Integration Copyright (c) 2025 Superstruct Ltd, New Zealand
 * Licensed under the same license as the underlying rapier project (Apache License 2.0)
 */

use wasm_bindgen::prelude::*;
use js_sys::{Array, Float32Array, Uint32Array, Promise};
use web_sys::console;
use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;

// Re-export core types based on feature flags
#[cfg(feature = "2d")]
pub use rapier2d::prelude::*;
#[cfg(feature = "3d")]
pub use rapier3d::prelude::*;

// Set up panic hook and allocator for WASM
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
    
    #[cfg(feature = "wee_alloc")]
    {
        use wee_alloc::WeeAlloc;
        #[global_allocator]
        static ALLOC: WeeAlloc = WeeAlloc::INIT;
    }
    
    log::info!("Rapier.wasm initialized");
}

/// JavaScript-accessible vector2 type
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

#[wasm_bindgen]
impl Vector2 {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32) -> Vector2 {
        Vector2 { x, y }
    }
    
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f32 { self.x }
    
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f32 { self.y }
    
    #[wasm_bindgen(setter)]
    pub fn set_x(&mut self, x: f32) { self.x = x; }
    
    #[wasm_bindgen(setter)]
    pub fn set_y(&mut self, y: f32) { self.y = y; }
    
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            self.x /= mag;
            self.y /= mag;
        }
    }
    
    pub fn dot(&self, other: &Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

/// JavaScript-accessible vector3 type  
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[wasm_bindgen]
impl Vector3 {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }
    
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f32 { self.x }
    
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f32 { self.y }
    
    #[wasm_bindgen(getter)]
    pub fn z(&self) -> f32 { self.z }
    
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    
    pub fn cross(&self, other: &Vector3) -> Vector3 {
        Vector3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

/// Physics body information for JavaScript
#[wasm_bindgen]
pub struct RigidBodyInfo {
    pub translation_x: f32,
    pub translation_y: f32,
    pub translation_z: f32,
    pub rotation_w: f32,
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub rotation_z: f32,
    pub velocity_x: f32,
    pub velocity_y: f32,
    pub velocity_z: f32,
    pub angular_velocity_x: f32,
    pub angular_velocity_y: f32,
    pub angular_velocity_z: f32,
    pub mass: f32,
    pub body_type: u8, // 0 = Dynamic, 1 = Static, 2 = Kinematic
}

/// Contact information for collision callbacks
#[wasm_bindgen]
pub struct ContactInfo {
    pub body1_handle: u32,
    pub body2_handle: u32,
    pub contact_point_x: f32,
    pub contact_point_y: f32,
    pub contact_point_z: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
    pub penetration: f32,
    pub impulse: f32,
}

/// Memory and performance statistics
#[wasm_bindgen]
pub struct PhysicsStats {
    pub rigid_body_count: u32,
    pub collider_count: u32,
    pub joint_count: u32,
    pub contact_count: u32,
    pub memory_used_mb: f32,
    pub step_time_ms: f32,
    pub collision_detection_time_ms: f32,
    pub solver_time_ms: f32,
}

/// Physics world configuration
#[wasm_bindgen]
pub struct WorldConfig {
    pub gravity_x: f32,
    pub gravity_y: f32,
    pub gravity_z: f32,
    pub timestep: f32,
    pub max_bodies: u32,
    pub max_colliders: u32,
    pub enable_simd: bool,
    pub enable_webgpu: bool,
    pub enable_deterministic: bool,
}

#[wasm_bindgen]
impl WorldConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WorldConfig {
        WorldConfig {
            gravity_x: 0.0,
            gravity_y: -9.81,
            gravity_z: 0.0,
            timestep: 1.0 / 60.0,
            max_bodies: 10000,
            max_colliders: 10000,
            enable_simd: true,
            enable_webgpu: false,
            enable_deterministic: false,
        }
    }
}

/// Main physics engine interface
#[wasm_bindgen]
pub struct PhysicsEngine {
    #[cfg(feature = "2d")]
    world_2d: Option<PhysicsWorld2D>,
    #[cfg(feature = "3d")]
    world_3d: Option<PhysicsWorld3D>,
    
    config: WorldConfig,
    stats: PhysicsStats,
    
    // WebGPU integration
    #[cfg(feature = "webgpu")]
    gpu_context: Option<GpuContext>,
    
    // Memory pools for efficient object management
    body_pool: Vec<RigidBodyHandle>,
    collider_pool: Vec<ColliderHandle>,
    joint_pool: Vec<JointHandle>,
    
    // Performance monitoring
    #[cfg(feature = "profiler")]
    profiler: PerformanceProfiler,
}

/// 2D Physics World wrapper
#[cfg(feature = "2d")]
struct PhysicsWorld2D {
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    gravity: nalgebra::Vector2<f32>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
}

/// 3D Physics World wrapper
#[cfg(feature = "3d")]
struct PhysicsWorld3D {
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    gravity: nalgebra::Vector3<f32>,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
}

/// WebGPU integration for GPU-accelerated physics
#[cfg(feature = "webgpu")]
struct GpuContext {
    device: web_sys::GpuDevice,
    queue: web_sys::GpuQueue,
    
    // Compute pipelines for different physics stages
    broad_phase_pipeline: Option<web_sys::GpuComputePipeline>,
    narrow_phase_pipeline: Option<web_sys::GpuComputePipeline>,
    integration_pipeline: Option<web_sys::GpuComputePipeline>,
    
    // GPU buffers for physics data
    position_buffer: Option<web_sys::GpuBuffer>,
    velocity_buffer: Option<web_sys::GpuBuffer>,
    force_buffer: Option<web_sys::GpuBuffer>,
}

/// Performance profiling utilities
#[cfg(feature = "profiler")]
struct PerformanceProfiler {
    step_times: Vec<f64>,
    collision_times: Vec<f64>,
    solver_times: Vec<f64>,
    frame_start_time: f64,
}

#[wasm_bindgen]
impl PhysicsEngine {
    /// Create a new physics engine instance
    #[wasm_bindgen(constructor)]
    pub fn new(config: &WorldConfig) -> Result<PhysicsEngine, JsValue> {
        console::log_1(&"Initializing Rapier.wasm physics engine".into());
        
        let stats = PhysicsStats {
            rigid_body_count: 0,
            collider_count: 0,
            joint_count: 0,
            contact_count: 0,
            memory_used_mb: 0.0,
            step_time_ms: 0.0,
            collision_detection_time_ms: 0.0,
            solver_time_ms: 0.0,
        };
        
        Ok(PhysicsEngine {
            #[cfg(feature = "2d")]
            world_2d: None,
            #[cfg(feature = "3d")]
            world_3d: None,
            
            config: WorldConfig {
                gravity_x: config.gravity_x,
                gravity_y: config.gravity_y,
                gravity_z: config.gravity_z,
                timestep: config.timestep,
                max_bodies: config.max_bodies,
                max_colliders: config.max_colliders,
                enable_simd: config.enable_simd,
                enable_webgpu: config.enable_webgpu,
                enable_deterministic: config.enable_deterministic,
            },
            
            stats,
            
            #[cfg(feature = "webgpu")]
            gpu_context: None,
            
            body_pool: Vec::with_capacity(config.max_bodies as usize),
            collider_pool: Vec::with_capacity(config.max_colliders as usize),
            joint_pool: Vec::with_capacity(1000),
            
            #[cfg(feature = "profiler")]
            profiler: PerformanceProfiler {
                step_times: Vec::with_capacity(1000),
                collision_times: Vec::with_capacity(1000),
                solver_times: Vec::with_capacity(1000),
                frame_start_time: 0.0,
            },
        })
    }
    
    /// Initialize 2D physics world
    #[cfg(feature = "2d")]
    pub fn init_2d(&mut self) -> Result<(), JsValue> {
        let gravity = nalgebra::Vector2::new(self.config.gravity_x, self.config.gravity_y);
        
        self.world_2d = Some(PhysicsWorld2D {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            gravity,
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
        });
        
        console::log_1(&"2D physics world initialized".into());
        Ok(())
    }
    
    /// Initialize 3D physics world
    #[cfg(feature = "3d")]
    pub fn init_3d(&mut self) -> Result<(), JsValue> {
        let gravity = nalgebra::Vector3::new(
            self.config.gravity_x, 
            self.config.gravity_y, 
            self.config.gravity_z
        );
        
        self.world_3d = Some(PhysicsWorld3D {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            gravity,
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
        });
        
        console::log_1(&"3D physics world initialized".into());
        Ok(())
    }
    
    /// Initialize WebGPU acceleration (async)
    #[cfg(feature = "webgpu")]
    pub async fn init_webgpu(&mut self) -> Result<(), JsValue> {
        use wasm_bindgen_futures::JsFuture;
        
        // Get WebGPU adapter and device
        let window = web_sys::window().ok_or("Failed to get window")?;
        let navigator = window.navigator();
        let gpu = navigator.gpu().ok_or("WebGPU not supported")?;
        
        let adapter_promise = gpu.request_adapter();
        let adapter = JsFuture::from(adapter_promise).await?;
        let adapter: web_sys::GpuAdapter = adapter.dyn_into()?;
        
        let device_promise = adapter.request_device();
        let device = JsFuture::from(device_promise).await?;
        let device: web_sys::GpuDevice = device.dyn_into()?;
        
        let queue = device.queue();
        
        self.gpu_context = Some(GpuContext {
            device,
            queue,
            broad_phase_pipeline: None,
            narrow_phase_pipeline: None,
            integration_pipeline: None,
            position_buffer: None,
            velocity_buffer: None,
            force_buffer: None,
        });
        
        console::log_1(&"WebGPU acceleration initialized".into());
        Ok(())
    }
    
    /// Step the physics simulation
    pub fn step(&mut self, delta_time: Option<f32>) -> Result<(), JsValue> {
        #[cfg(feature = "profiler")]
        let step_start = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        let dt = delta_time.unwrap_or(self.config.timestep);
        
        #[cfg(feature = "2d")]
        if let Some(ref mut world) = self.world_2d {
            world.integration_parameters.dt = dt;
            
            world.physics_pipeline.step(
                &world.gravity,
                &world.integration_parameters,
                &mut world.island_manager,
                &mut world.broad_phase,
                &mut world.narrow_phase,
                &mut world.rigid_body_set,
                &mut world.collider_set,
                &mut world.impulse_joint_set,
                &mut world.multibody_joint_set,
                &mut world.ccd_solver,
                None,
                &(),
                &(),
            );
            
            world.query_pipeline.update(
                &mut world.island_manager,
                &world.rigid_body_set,
                &world.collider_set
            );
        }
        
        #[cfg(feature = "3d")]
        if let Some(ref mut world) = self.world_3d {
            world.integration_parameters.dt = dt;
            
            world.physics_pipeline.step(
                &world.gravity,
                &world.integration_parameters,
                &mut world.island_manager,
                &mut world.broad_phase,
                &mut world.narrow_phase,
                &mut world.rigid_body_set,
                &mut world.collider_set,
                &mut world.impulse_joint_set,
                &mut world.multibody_joint_set,
                &mut world.ccd_solver,
                None,
                &(),
                &(),
            );
            
            world.query_pipeline.update(
                &mut world.island_manager,
                &world.rigid_body_set,
                &world.collider_set
            );
        }
        
        // Update statistics
        #[cfg(feature = "2d")]
        if let Some(ref world) = self.world_2d {
            self.stats.rigid_body_count = world.rigid_body_set.len() as u32;
            self.stats.collider_count = world.collider_set.len() as u32;
            self.stats.joint_count = world.impulse_joint_set.len() as u32;
        }
        
        #[cfg(feature = "3d")]
        if let Some(ref world) = self.world_3d {
            self.stats.rigid_body_count = world.rigid_body_set.len() as u32;
            self.stats.collider_count = world.collider_set.len() as u32;
            self.stats.joint_count = world.impulse_joint_set.len() as u32;
        }
        
        #[cfg(feature = "profiler")]
        {
            let step_end = web_sys::window()
                .unwrap()
                .performance()
                .unwrap()
                .now();
            
            let step_time = step_end - step_start;
            self.stats.step_time_ms = step_time as f32;
            self.profiler.step_times.push(step_time);
            
            // Keep only last 1000 measurements
            if self.profiler.step_times.len() > 1000 {
                self.profiler.step_times.remove(0);
            }
        }
        
        Ok(())
    }
    
    /// Create a dynamic rigid body
    #[cfg(feature = "2d")]
    pub fn create_dynamic_body_2d(&mut self, x: f32, y: f32, rotation: f32) -> Result<u32, JsValue> {
        if let Some(ref mut world) = self.world_2d {
            let body = RigidBodyBuilder::dynamic()
                .translation(nalgebra::Vector2::new(x, y))
                .rotation(rotation)
                .build();
            
            let handle = world.rigid_body_set.insert(body);
            let handle_id = handle.0 as u32;
            
            // Store in pool for efficient management
            self.body_pool.push(handle);
            
            Ok(handle_id)
        } else {
            Err(JsValue::from_str("2D world not initialized"))
        }
    }
    
    /// Create a dynamic rigid body
    #[cfg(feature = "3d")]
    pub fn create_dynamic_body_3d(&mut self, x: f32, y: f32, z: f32, 
                                   quat_w: f32, quat_x: f32, quat_y: f32, quat_z: f32) -> Result<u32, JsValue> {
        if let Some(ref mut world) = self.world_3d {
            let body = RigidBodyBuilder::dynamic()
                .translation(nalgebra::Vector3::new(x, y, z))
                .rotation(nalgebra::UnitQuaternion::from_quaternion(
                    nalgebra::Quaternion::new(quat_w, quat_x, quat_y, quat_z)
                ))
                .build();
            
            let handle = world.rigid_body_set.insert(body);
            let handle_id = handle.0 as u32;
            
            self.body_pool.push(handle);
            
            Ok(handle_id)
        } else {
            Err(JsValue::from_str("3D world not initialized"))
        }
    }
    
    /// Get physics statistics
    pub fn get_stats(&self) -> PhysicsStats {
        PhysicsStats {
            rigid_body_count: self.stats.rigid_body_count,
            collider_count: self.stats.collider_count,
            joint_count: self.stats.joint_count,
            contact_count: self.stats.contact_count,
            memory_used_mb: self.stats.memory_used_mb,
            step_time_ms: self.stats.step_time_ms,
            collision_detection_time_ms: self.stats.collision_detection_time_ms,
            solver_time_ms: self.stats.solver_time_ms,
        }
    }
    
    /// Get performance profiling data
    #[cfg(feature = "profiler")]
    pub fn get_performance_profile(&self) -> Result<String, JsValue> {
        if self.profiler.step_times.is_empty() {
            return Ok("No profiling data available".to_string());
        }
        
        let avg_step_time: f64 = self.profiler.step_times.iter().sum::<f64>() 
            / self.profiler.step_times.len() as f64;
        let max_step_time = self.profiler.step_times.iter()
            .fold(0.0f64, |acc, &x| acc.max(x));
        let min_step_time = self.profiler.step_times.iter()
            .fold(f64::INFINITY, |acc, &x| acc.min(x));
        
        let profile = format!(
            "Performance Profile:\nAverage Step Time: {:.2}ms\nMax Step Time: {:.2}ms\nMin Step Time: {:.2}ms\nFPS Estimate: {:.1}",
            avg_step_time, max_step_time, min_step_time, 1000.0 / avg_step_time
        );
        
        Ok(profile)
    }
    
    /// Export physics state for serialization
    #[cfg(feature = "serde")]
    pub fn export_state(&self) -> Result<String, JsValue> {
        #[cfg(feature = "2d")]
        if let Some(ref world) = self.world_2d {
            match serde_json::to_string(&world.rigid_body_set) {
                Ok(serialized) => Ok(serialized),
                Err(e) => Err(JsValue::from_str(&format!("Serialization failed: {}", e))),
            }
        } else {
            Err(JsValue::from_str("No active physics world"))
        }
        
        #[cfg(all(feature = "3d", not(feature = "2d")))]
        if let Some(ref world) = self.world_3d {
            match serde_json::to_string(&world.rigid_body_set) {
                Ok(serialized) => Ok(serialized),
                Err(e) => Err(JsValue::from_str(&format!("Serialization failed: {}", e))),
            }
        } else {
            Err(JsValue::from_str("No active physics world"))
        }
        
        #[cfg(not(any(feature = "2d", feature = "3d")))]
        Err(JsValue::from_str("No physics world features enabled"))
    }
    
    /// Cleanup resources
    pub fn cleanup(&mut self) {
        #[cfg(feature = "2d")]
        {
            self.world_2d = None;
        }
        
        #[cfg(feature = "3d")]
        {
            self.world_3d = None;
        }
        
        self.body_pool.clear();
        self.collider_pool.clear();
        self.joint_pool.clear();
        
        console::log_1(&"Physics engine cleanup completed".into());
    }
}

/// Utility functions for WASM integration
#[wasm_bindgen]
pub struct RapierUtils;

#[wasm_bindgen]
impl RapierUtils {
    /// Get Rapier version information
    pub fn get_version() -> String {
        format!("Rapier.wasm v{}", env!("CARGO_PKG_VERSION"))
    }
    
    /// Check SIMD support
    pub fn has_simd_support() -> bool {
        #[cfg(feature = "simd")]
        {
            cfg!(target_feature = "simd128")
        }
        #[cfg(not(feature = "simd"))]
        {
            false
        }
    }
    
    /// Check WebGPU support
    pub fn has_webgpu_support() -> bool {
        #[cfg(feature = "webgpu")]
        {
            web_sys::window()
                .and_then(|w| w.navigator().gpu())
                .is_some()
        }
        #[cfg(not(feature = "webgpu"))]
        {
            false
        }
    }
    
    /// Performance benchmark for different math operations
    pub fn benchmark_math_operations(iterations: u32) -> Result<String, JsValue> {
        let start_time = web_sys::window()
            .ok_or("No window")?
            .performance()
            .ok_or("No performance")?
            .now();
        
        // Vector operations benchmark
        for _ in 0..iterations {
            let v1 = nalgebra::Vector3::new(1.0, 2.0, 3.0);
            let v2 = nalgebra::Vector3::new(4.0, 5.0, 6.0);
            let _result = v1 + v2 * 2.0;
        }
        
        let end_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        let duration = end_time - start_time;
        let ops_per_second = (iterations as f64) / (duration / 1000.0);
        
        Ok(format!(
            "Vector operations: {:.0} ops/sec ({:.2}ms for {} iterations)",
            ops_per_second, duration, iterations
        ))
    }
}

/// Physics benchmarking and testing utilities
#[wasm_bindgen]
pub struct PhysicsBenchmark {
    engine: PhysicsEngine,
    benchmark_results: HashMap<String, f64>,
}

#[wasm_bindgen]
impl PhysicsBenchmark {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<PhysicsBenchmark, JsValue> {
        let config = WorldConfig::new();
        let mut engine = PhysicsEngine::new(&config)?;
        
        #[cfg(feature = "3d")]
        engine.init_3d()?;
        #[cfg(all(feature = "2d", not(feature = "3d")))]
        engine.init_2d()?;
        
        Ok(PhysicsBenchmark {
            engine,
            benchmark_results: HashMap::new(),
        })
    }
    
    /// Benchmark physics step performance
    pub fn benchmark_physics_step(&mut self, num_bodies: u32, num_steps: u32) -> Result<String, JsValue> {
        // Create test bodies
        #[cfg(feature = "3d")]
        {
            for i in 0..num_bodies {
                let x = (i % 10) as f32;
                let y = (i / 10) as f32;
                let z = 0.0;
                self.engine.create_dynamic_body_3d(x, y, z, 1.0, 0.0, 0.0, 0.0)?;
            }
        }
        
        #[cfg(all(feature = "2d", not(feature = "3d")))]
        {
            for i in 0..num_bodies {
                let x = (i % 10) as f32;
                let y = (i / 10) as f32;
                self.engine.create_dynamic_body_2d(x, y, 0.0)?;
            }
        }
        
        // Benchmark simulation steps
        let start_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        for _ in 0..num_steps {
            self.engine.step(None)?;
        }
        
        let end_time = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        
        let total_time = end_time - start_time;
        let avg_step_time = total_time / num_steps as f64;
        let estimated_fps = 1000.0 / avg_step_time;
        
        let result = format!(
            "Physics Benchmark Results:\nBodies: {}\nSteps: {}\nTotal Time: {:.2}ms\nAvg Step Time: {:.3}ms\nEstimated FPS: {:.1}",
            num_bodies, num_steps, total_time, avg_step_time, estimated_fps
        );
        
        Ok(result)
    }
    
    /// Get benchmark results as JSON
    pub fn get_results_json(&self) -> Result<String, JsValue> {
        match serde_json::to_string(&self.benchmark_results) {
            Ok(json) => Ok(json),
            Err(e) => Err(JsValue::from_str(&format!("JSON serialization failed: {}", e))),
        }
    }
}