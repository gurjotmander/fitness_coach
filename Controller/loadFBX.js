import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';
import { FBXLoader } from '/Controller/loaders/FBXLoader.js';

let scene, camera, renderer, mixer;

function init3D() {
  // Create scene
  scene = new THREE.Scene();

  // Set up camera
  camera = new THREE.PerspectiveCamera(75, fbxCanvas.width / fbxCanvas.height, 0.1, 1000);
  camera.position.set(0, 2, 5);

  // Create and attach WebGLRenderer to fbxCanvas
  renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('fbxCanvas'), alpha: true });
  renderer.setSize(fbxCanvas.width, fbxCanvas.height);

  // Lighting
  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(5, 10, 7.5);
  scene.add(light);

  // Load FBX
  const loader = new FBXLoader();
  loader.load('/View/assets/Air Squat.fbx', (object) => {
    mixer = new THREE.AnimationMixer(object);
    const action = mixer.clipAction(object.animations[0]);
    action.play();

    object.scale.set(0.01, 0.01, 0.01);
    scene.add(object);
  });

  animate3D();
}

function animate3D() {
  requestAnimationFrame(animate3D);

  if (mixer) mixer.update(0.01);

  renderer.render(scene, camera);
}

// Initialize the 3D scene
init3D();