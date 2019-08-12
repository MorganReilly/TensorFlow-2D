/**
 * Initial Welcome Message
 */
console.log('Hello TensorFlow');

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataReq.json();
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
        .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;
}

/**
 * Plotting data in scatter plot to see how it looks
 */
async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));

    tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );

    // More code will be added below
}

document.addEventListener('DOMContentLoaded', run);

/**
 * Defining model architecture
 */
function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    /** 
     * Add a single hidden layer
     * Where:
     * dense -- layer type that multiplies its inputs by a matrix,
     *          then adds a number (bias) to the result. 
     * inputShape -- horsepower of given car.
     * units -- sets how big the weight matrix will be in the layer.
    */
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

    // Add an output layer
    model.add(tf.layers.dense({ units: 1, useBias: true }));

    return model;
}

/**
 * Creating an instance of the model 
 * and show summary of layers on webpage
 */
const model = createModel();
tfvis.show.modelSummary({ name: 'Model Summary' }, model);

/**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.

    return tf.tidy(() => {
        /**
         * Step 1. Shuffle the data
         * Randomize the order of the examples we will feed to the training algorithm.
         */
        tf.util.shuffle(data);

        /** 
         * Step 2. Convert data to Tensor
         * Make two arrays, one for input examples (horsepower entries)
         * another for the true output values (known as labels in machine learning)
        */
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg);

        /**
         * Converting each array data into 2D tensor
         */
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        /**
         * Step 3. Normalize the data to the range 0 - 1 using min-max scaling
         * This is important as the intervals of many machine models built with TensorFlow.js are desinged to
         * work with numbers that are not too big. 
         * Common ranges to normalize data include 0-1. -1 - 1
        */
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}