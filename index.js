
async function GenerateNegatives() {
    const model = tf.sequential()
    model.add(tf.layers.dense({
        inputShape: [1],
        units: 1
    }))

    model.compile({
        loss: "meanSquaredError",
        optimizer: "sgd",
        matrix: ["accuracy"]
    })

    const data = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [8, 1])
    const labels = tf.tensor2d([-1, -2, -3, -4, -5, -6, -7, -8], [8, 1])

    await model.fit(data, labels, { epochs: 1000 })
    document.getElementById("result").innerHTML = model.predict(tf.tensor2d([13], [1, 1]))
}

GenerateNegatives();