const status = document.getElementById("status");
const download = document.getElementById("download");
const trainButton = document.getElementById("train");
const testButton = document.getElementById("test");
const loadWeightsButton = document.getElementById("loadWeights");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const canvas = document.getElementById("canvas");
const prediction = document.getElementById("prediction");

canvas.style.backgroundColor = "black";
const ctx = canvas.getContext("2d");
