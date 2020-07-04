import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis"

import data from "./data"

interface Data {
  tradeSide: number,
  tb1_open: number,
  tb1_high: number,
  tb1_low: number,
  tb1_close: number
  tb2_open: number,
  tb2_high: number,
  tb2_low: number,
  tb2_close: number
  opp_tradeSide: number,
  opp_enter: number,
  opp_high: number,
  opp_low: number
}
async function getData(): Promise<Data[]> {
    const flattened = data.map(d => ({
      tradeSide: d.tradeSide === "BUY" ? 1 : (d.tradeSide === "SELL" ? -1 : 0),
      tb1_open: d.trendbars[0].open,
      tb1_high: d.trendbars[0].high,
      tb1_low: d.trendbars[0].low,
      tb1_close: d.trendbars[0].close,
      tb2_open: d.trendbars[1].open,
      tb2_high: d.trendbars[1].high,
      tb2_low: d.trendbars[1].low,
      tb2_close: d.trendbars[1].close,
      opp_tradeSide: d.oppurtunities[0].orderType === "BUY" ? 1 : (d.oppurtunities[0].orderType === "SELL" ? -1 : 0),
      opp_enter: d.oppurtunities[0].enter,
      opp_high: d.oppurtunities[0].high,
      opp_low: d.oppurtunities[0].low
    }))
    return flattened;
}

function normalize(d: Data): Data {
  const max = Math.max(          d.tb1_open,    d.tb1_high,    d.tb1_low,    d.tb1_close,    d.tb2_open,    d.tb2_high,    d.tb2_low,    d.tb2_close,    d.tradeSide);
  const min = Math.min(          d.tb1_open,    d.tb1_high,    d.tb1_low,    d.tb1_close,    d.tb2_open,    d.tb2_high,    d.tb2_low,    d.tb2_close,    d.tradeSide);
  const range = max - min;
  return {
    tradeSide: d.tradeSide,
    tb1_open: (d.tb1_open - min) / range,
    tb1_high: (d.tb1_high - min) / range,
    tb1_low: (d.tb1_low - min) / range,
    tb1_close: (d.tb1_close - min) / range,
    tb2_open: (d.tb2_open - min) / range,
    tb2_high: (d.tb2_high - min) / range,
    tb2_low: (d.tb2_low - min) / range,
    tb2_close: (d.tb2_close - min) / range,

    opp_tradeSide: d.opp_tradeSide, 
    opp_enter: (d.opp_enter - min) / range,
    opp_high: (d.opp_high - min) / range,
    opp_low: (d.opp_low - min) / range
  }
}
function unnormalize(d: Data, orig: Data): Data {
  const max = Math.max(          orig.tb1_open,    orig.tb1_high,    orig.tb1_low,    orig.tb1_close,    orig.tb2_open,    orig.tb2_high,    orig.tb2_low,    orig.tb2_close,    orig.tradeSide);
  const min = Math.min(          orig.tb1_open,    orig.tb1_high,    orig.tb1_low,    orig.tb1_close,    orig.tb2_open,    orig.tb2_high,    orig.tb2_low,    orig.tb2_close,    orig.tradeSide);
  const range = max - min;
  return {
    tradeSide: d.tradeSide,
    tb1_open: (d.tb1_open * range) + min,
    tb1_high: (d.tb1_high * range) + min,
    tb1_low: (d.tb1_low * range) + min,
    tb1_close: (d.tb1_close * range) + min,
    tb2_open: (d.tb2_open * range) + min,
    tb2_high: (d.tb2_high * range) + min,
    tb2_low: (d.tb2_low * range) + min,
    tb2_close: (d.tb2_close * range) + min,

    opp_tradeSide: d.opp_tradeSide, 
    opp_enter: (d.opp_enter * range) + min,
    opp_high: (d.opp_high * range) + min,
    opp_low: (d.opp_low * range) + min,
  }
}

function createModel(): tf.LayersModel {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [9], units: 100, useBias: true }));
    // model.add(tf.layers.dense({ units: 100, activation: 'sigmoid', useBias: true }));
    model.add(tf.layers.dense({ units: 1000, activation: 'tanh', useBias: false }));
    // model.add(tf.layers.dense({ units: 50, activation: 'softmax', useBias: true }));

    // Add an output layer
    model.add(tf.layers.dense({ units: 4, activation: undefined, useBias: false }));

    return model;
}

function convertToTensor(data: Data[]) {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data    
        const tmp = [...data];
        tf.util.shuffle(tmp);
        
        // Step 2. Normalize the data
        const normalized = tmp.map(d => normalize(d));

        // Step 3. Convert data to Tensor
        const inputs = normalized.map(d => ([
          d.tb1_open,
          d.tb1_high,
          d.tb1_low,
          d.tb1_close,
          d.tb2_open,
          d.tb2_high,
          d.tb2_low,
          d.tb2_close,
          d.tradeSide
        ]));
        const labels = normalized.map(d => ([
          d.opp_tradeSide, 
          d.opp_enter, 
          d.opp_high,
          d.opp_low
        ]));

        return {
            inputs: tf.tensor2d(inputs, [inputs.length, 9]),
            labels: tf.tensor2d(labels, [labels.length, 4])
        }
    });
}

async function trainModel(model: tf.LayersModel, inputs: tf.Tensor, labels: tf.Tensor) {
  // Prepare the model for training.  
  model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 150;

  return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
          { name: 'Training Performance' },
          ['loss', 'mse'],
          { height: 200, callbacks: ['onEpochEnd'], yAxisDomain: [0,0.2] }
      )
  });
}

async function testModel(model: tf.LayersModel, sample: Data[]) {
  tf.tidy(() => {
    console.log("1")
    const normalized = sample.map(d => normalize(d));
    console.log("2")
    const inputs = normalized.map(d => ([
      d.tb1_open,
      d.tb1_high,
      d.tb1_low,
      d.tb1_close,
      d.tb2_open,
      d.tb2_high,
      d.tb2_low,
      d.tb2_close,
      d.tradeSide
    ]));
    console.log("3")
    const inputsTensor = tf.tensor2d(inputs, [inputs.length, 9]);
    console.log("4")
      const labels = model.predict(inputsTensor) as tf.Tensor;
      const predictions = labels.reshape([sample.length, 4]).arraySync() as number[][]
      console.log("----", predictions.map((p,i) => unnormalize({
        ...normalized[i],
        opp_tradeSide: p[0], 
        opp_enter: p[1], 
        opp_high: p[2],
        opp_low: p[3]
      }, sample[i])));

      // Un-normalize the data
      // return [unNormXs.dataSync(), unNormPreds.dataSync()];
      return []
  });


  // const predictedPoints = Array.from(xs).map((val, i) => {
  //     return { x: val, y: preds[i] }
  // });

  // const originalPoints = inputData.map(d => ({
  //     x: d.horsepower, y: d.mpg,
  // }));


  // await tfvis.render.scatterplot(
  //     { name: 'Model Predictions vs Original Data' },
  //     { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
  //     {
  //         xLabel: 'Horsepower',
  //         yLabel: 'MPG',
  //         height: 300
  //     }
  // );
}

async function main() {
    const data = await getData();

    // const model = createModel();
    // tfvis.show.modelSummary({ name: 'Model Summary' }, model);

    // const tensorData = convertToTensor(data);
    // const { inputs, labels } = tensorData;
    // await trainModel(model, inputs, labels);
    // console.log('Done Training');
    // await model.save(`localstorage://my-model-${Date.now()}`)

 
    const modelLoaded = await tf.loadLayersModel("localstorage://my-model-1593712475905");
    await testModel(modelLoaded, data);
}

document.addEventListener('DOMContentLoaded', main);
