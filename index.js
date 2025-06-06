// index.js

// Using 'import' statements for ES Modules
import express from 'express';
import path from 'path';
import fs from 'fs';
import fetch from 'node-fetch';
import sharp from 'sharp';
// Remove the separate TensorFlow.js import
// import * as tf from '@tensorflow/tfjs-node'; // REMOVED

// Import face-api.js and its dependencies for Node.js
import * as faceapi from 'face-api.js';
import '@tensorflow/tfjs-node'; // Import tfjs-node but don't assign to variable
import { Canvas, Image, ImageData } from 'canvas';

// In ES Modules, __dirname is not directly available. We need to create it.
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Patch Canvas for face-api.js
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const port = 3000;

// Middleware to parse JSON and url-encoded bodies
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Path to store models
const MODEL_URL = path.join(__dirname, 'models');

let modelsLoaded = false;

// Function to load face-api.js models
async function loadModels() {
  try {
    console.log('Loading face-api.js models...');

    console.log('Loading face-api.js SSD MobileNet V1 model...');
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
    console.log('SSD MobileNet V1 model loaded.');

    console.log('Loading face-api.js Face Landmark 68 Net model...');
    await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
    console.log('Face Landmark 68 Net model loaded.');

    console.log('Loading face-api.js Face Recognition Net model...');
    await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);
    console.log('Face Recognition Net model loaded.');

    console.log('All face-api.js models loaded successfully.');
    modelsLoaded = true;
  } catch (error) {
    console.error('Failed to load models:', error);
    process.exit(1);
  }
}

async function loadImage(imagePath) {
  let buffer;

  if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
    const response = await fetch(imagePath);
    const arrayBuffer = await response.arrayBuffer();
    buffer = Buffer.from(arrayBuffer);
  } else {
    buffer = fs.readFileSync(path.resolve(imagePath));
  }

  const image = await sharp(buffer)
    .resize(512) // Optional: normalize image size
    .toBuffer();

  const img = new Image();
  img.src = image;

  return img;
}

// Endpoint to compare two images
app.post('/compare-faces', async (req, res) => {
  if (!modelsLoaded) {
    return res
      .status(503)
      .json({ error: 'Models are still loading. Please try again shortly.' });
  }

  const { image1Path, image2Path, threshold = 0.6 } = req.body;

  if (!image1Path || !image2Path) {
    return res.status(400).json({
      error:
        'Please provide both image1Path and image2Path in the request body.',
    });
  }

  try {
    const img1 = await loadImage(image1Path);
    const img2 = await loadImage(image2Path);

    const detections1 = await faceapi
      .detectAllFaces(img1)
      .withFaceLandmarks()
      .withFaceDescriptors();

    console.log(detections1);
    console.log('Face detection complete for image 1.');

    console.log('Attempting face detection for image 2...');
    const detections2 = await faceapi
      .detectAllFaces(img2)
      .withFaceLandmarks()
      .withFaceDescriptors();
    console.log('Face detection complete for image 2.');

    console.log(
      `Detected ${detections1.length} faces in image 1 and ${detections2.length} faces in image 2.`,
    );

    if (detections1.length === 0 || detections2.length === 0) {
      return res.status(400).json({
        match: false,
        message: 'Could not detect faces in one or both images.',
      });
    }

    const faceMatcher = new faceapi.FaceMatcher(detections2);

    let bestMatch = null;
    let bestDistance = Infinity;

    detections1.forEach((detection1) => {
      const match = faceMatcher.findBestMatch(detection1.descriptor);
      if (match.distance < bestDistance) {
        bestDistance = match.distance;
        bestMatch = match;
      }
    });

    const isMatch = bestDistance < threshold;

    res.json({
      match: isMatch,
      distance: bestDistance,
      threshold: threshold,
      message: isMatch ? 'Faces match!' : 'Faces do not match.',
      details: {
        facesDetectedImage1: detections1.length,
        facesDetectedImage2: detections2.length,
      },
    });
  } catch (error) {
    console.error('Error during face comparison:', error);
    res.status(500).json({
      error: `Internal server error during face comparison: ${error.message}`,
    });
  }
});

// Start the server after models are loaded
loadModels().then(() => {
  app.listen(port, () => {
    console.log(`Face match app listening at http://localhost:${port}`);
    console.log(
      `Send POST request to http://localhost:${port}/compare-faces with image1Path and image2Path in body.`,
    );
    console.log(
      `Example for local files: curl -X POST -H "Content-Type: application/json" -d '{"image1Path": "person1.jpg", "image2Path": "person1_another_pose.jpg"}' http://localhost:3000/compare-faces`,
    );
    console.log(
      `Example for URLs: curl -X POST -H "Content-Type: application/json" -d '{"image1Path": "https://example.com/image1.jpg", "image2Path": "https://example.com/image2.jpg"}' http://localhost:3000/compare-faces`,
    );
  });
});
