const inputnodes = 784;
const hiddennodes = 100;
const outputnodes = 10;
const learningrate = 0.2;
let iter = 0;
const iterations = 5;

const trainingDataPath = "./mnist/mnist_train.csv";
const testDataPath = "./mnist/mnist_test.csv";
const weightsFilename = "weights.json";
const savedWeightsPath = `./dist/${weightsFilename}`;

const trainingData = [];
const trainingLabels = [];
const testData = [];
const testLabels = [];
const savedWeights = {};

const printSteps = 1;

let NN;

window.onload = async () => {
  NN = new NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate);

  trainButton.disabled = true;
  testButton.disabled = true;
  loadWeightsButton.disabled = true;

  status.innerHTML = "Loading the data sets. Please wait ...<br>";

  const trainCSV = await loadData(trainingDataPath, "CSV");

  if (trainCSV) {
    prepareData(trainCSV, trainingData, trainingLabels);
    status.innerHTML += "Training data successfully loaded...<br>";
  }

  const testCSV = await loadData(testDataPath, "CSV");

  if (testCSV) {
    prepareData(testCSV, testData, testLabels);
    status.innerHTML += "Test data successfully loaded...<br>";
  }

  if (!trainCSV || !testCSV) {
    status.innerHTML +=
      "Error loading train/test data set. Please check your file path! If you run this project locally, it needs to be on a local server.";
    return;
  }

  trainButton.disabled = false;
  testButton.disabled = false;

  const weightsJSON = await loadData(savedWeightsPath, "JSON");

  if (weightsJSON) {
    savedWeights.wih = weightsJSON.wih;
    savedWeights.who = weightsJSON.who;
    loadWeightsButton.disabled = false;
  }

  status.innerHTML += "Ready.<br><br>";
};

async function loadData(path, type) {
  try {
    const result = await fetch(path, {
      mode: "no-cors",
    });

    switch (type) {
      case "CSV":
        return await result.text();
      case "JSON":
        return await result.json();
      default:
        return false;
    }
  } catch {
    return false;
  }
}

function prepareData(rawData, target, labels) {
  rawData = rawData.split("\n");
  rawData.pop();

  rawData.forEach((current) => {
    let sample = current.split(",").map((x) => +x);

    labels.push(sample[0]);
    sample.shift();

    sample = NeuralNetwork.normalizeData(sample);

    target.push(sample);
  });
}

function train() {
  trainButton.disabled = true;
  testButton.disabled = true;
  loadWeightsButton.disabled = true;
  download.innerHTML = "";

  if (iter < iterations) {
    iter++;
    status.innerHTML += "Starting training ...<br>";
    status.innerHTML += "Iteration " + iter + " of " + iterations + "<br>";

    trainingData.forEach((current, index) => {
      setTimeout(() => {
        const label = trainingLabels[index];
        const oneHotLabel = Array(10).fill(0);
        oneHotLabel[label] = 0.99;

        NN.train(current, oneHotLabel);

        if (index > 0 && !((index + 1) % printSteps)) {
          status.innerHTML += `finished  ${index + 1}  samples ... <br>`;
        }

        if (index === trainingData.length - 1) {
          status.innerHTML += `Loss:  ${
            math.sum(NN.cache.loss) / trainingData.length
          }<br><br>`;
          NN.cache.loss = [];

          test("", true);
        }
      }, 0);
    });
  }
}

function test(_, inTraining = false) {
  trainButton.disabled = true;
  testButton.disabled = true;
  loadWeightsButton.disabled = true;

  status.innerHTML += "Starting testing ...<br>";

  let correctPredicts = 0;
  testData.forEach((current, index) => {
    setTimeout(() => {
      const actual = testLabels[index];

      const predict = formatPrediction(NN.predict(current));
      predict === actual ? correctPredicts++ : null;

      if (index > 0 && !((index + 1) % printSteps)) {
        status.innerHTML += " finished " + (index + 1) + " samples ...<br>";
      }

      if (index >= testData.length - 1) {
        status.innerHTML +=
          "Accuracy: " +
          Math.round((correctPredicts / testData.length) * 100) +
          " %<br><br>";

        if (iter + 1 > iterations) {
          createDownloadLink();
          enableAllButtons();
          status.innerHTML += "Finished training.<br><br>";
          iter = 0;
        } else if (inTraining) {
          train();
        } else {
          enableAllButtons();
        }
      }
    }, 0);
  });
}

function predict() {
  const tempCanvas = document.createElement("canvas");
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(canvas, 0, 0, 250, 250, 0, 0, 28, 28);

  const img = tempCtx.getImageData(0, 0, 28, 28);

  let sample = [];
  for (let i = 0, j = 0; i < img.data.length; i += 4, j++) {
    sample[j] = (img.data[i] + img.data[i + 1] + img.data[i + 2]) / 3;
  }

  img.data = NeuralNetwork.normalizeData(img.data);

  prediction.innerHTML = formatPrediction(NN.predict(sample));
}

function formatPrediction(prediction) {
  const flattened = prediction.toArray().map((x) => x[0]);

  return flattened.indexOf(Math.max(...flattened));
}

function loadWeights() {
  NN.wih = savedWeights.wih;
  NN.who = savedWeights.who;
  status.innerHTML += "Weights successfully loaded.";
}

function createDownloadLink() {
  const wih = NN.wih.toArray();
  const who = NN.who.toArray();
  const weights = { wih, who };
  download.innerHTML = `<a download="${weightsFilename}" id="downloadLink" href="data:text/json;charset=utf-8,${encodeURIComponent(
    JSON.stringify(weights)
  )}">Download model weights</a>`;
}

function enableAllButtons() {
  trainButton.disabled = false;
  testButton.disabled = false;
  loadWeightsButton.disabled = false;
}
