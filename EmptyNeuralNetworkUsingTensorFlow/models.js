const model = tf.sequential()

const configHidden = {
    units: 4 ,
    inputShape: [2] ,
    activation: 'sigmoid'
}
const hidden = tf.layers.dense(configHidden);
model.add(hidden)

const configOutput = {
    units: 1 ,
    activation: 'sigmoid'
}

const output = tf.layers.dense(configOutput)
model.add(output);

model.compile({optimizer: 'sgd', loss: 'meanSquaredError'})

const xs = tf.tensor2d([
    [0 , 0] ,
    [0.5 , 0.25] ,
    [1 , 1] 
])

const ys = tf.tensor2d([
    [1] ,
    [0.5] ,
    [0] 
])

train().then(() => {
    console.log("Training is complete ...")
    let outputs = model.predict(xs)
    outputs.print()
});

async function train (){
    for(let i = 0 ; i < 10000 ; i++){
        const response = await model.fit(xs,ys)
        console.log(response.history.loss[0])
    }
    
}

