
// import * as tf from '@tensorflow/tfjs';
const RAVELED_RES = 1024;

/*
These layers exist in the original model, but are not supported by
TensorflowJS. The first one can be replaced by preprocessing the
pixels in javascript, and the second one can be omitted entirely.
We'll therefore just make some dummy layer classes to substitute
for model's preprocessing layers.
*/
class Rescaling extends tf.layers.Layer {
  constructor() { super({}); }
  call(inputs, kwargs) { return inputs; }
  static get className() { return 'Rescaling'; }
}
tf.serialization.registerClass(Rescaling);

class RandomContrast extends tf.layers.Layer {
  constructor() { super({}); }
  call(inputs, kwargs) { return inputs; }
  static get className() { return 'RandomContrast'; }
}
tf.serialization.registerClass(RandomContrast);

async function init() {
  const model = await tf.loadLayersModel('tangler.tfjs/model.json');

  const video = document.getElementById("stream");
  const ctx = document.getElementById("webcam").getContext("2d");
  const raveledCtx = document.getElementById("raveled").getContext("2d");
  raveledCtx.strokeStyle = '#000';
  raveledCtx.lineWidth = RAVELED_RES*40e-6;
  const tangledCtx = document.getElementById("tangled").getContext("2d");
  let imgData = tangledCtx.getImageData(0,0,tangledCtx.canvas.width,tangledCtx.canvas.height);

  if (navigator.mediaDevices.getUserMedia) {
    let raf, crop;
    navigator.mediaDevices.getUserMedia({
      video: true
    }).then(function (stream) {
      console.log("Streaming", stream);
      video.srcObject = stream;
      video.onplaying = function() {
        crop = {w:video.videoHeight, h:video.videoHeight,
                x:(video.videoWidth - video.videoHeight) / 2, y:0};
        raf = requestAnimationFrame(loop);
      }
      video.onpause = function(){
        cancelAnimationFrame(raf);
      }
    }).catch(function (error) {
      console.error("Something went wrong!",error);
    });

    let busy_lock = false;
    function loop() {
      ctx.drawImage(video, crop.x, crop.y, crop.w, crop.h, 0, 0, ctx.canvas.width, ctx.canvas.height);
      let iData = ctx.getImageData(0,0,ctx.canvas.width,ctx.canvas.height);
      for (let idx=0; idx<iData.data.length/4; idx++) {
        let x = (idx/256)/127-1;
        let y = (idx%256)/127-1;
        if (x*x + y*y > 1) {
          iData.data[4*idx] = 127;
          iData.data[4*idx+1] = 127;
          iData.data[4*idx+2] = 127;
          iData.data[4*idx+3] = 255;
        }
      }
      ctx.putImageData(iData,0,0);
      iData = null;
      if (busy_lock === false) {
        busy_lock = true;
        inference().then(() => { busy_lock = false; }).catch(() => {busy_lock = false;});
      }
      raf = requestAnimationFrame(loop);
    }

    async function inference() {
      tf.engine().startScope();
      const offset = tf.scalar(127.5);

      let predicted = await model.predict(
        tf.cast(tf.browser.fromPixels(ctx.canvas), 'float32') // capture frame
        .sub(offset).div(offset) // Normalize to [-1,1]
        .mean(2).expandDims(0).expandDims(-1) // Convert to grayscale
      ).data();

      let resampled = predicted.map((x,idx) => {
        let result = 0;
        let min=-2.5,max=0.5
        for (let i=min; i<max; i+=(max-min)/15) {
          result += (x > i)*(Math.random()<1/20);
        }
        return result;
      });

      // Iterate through the resampled prediction, and
      // render both the raw matrix pixels, and the
      // raveled line drawing.
      let path = new Path2D();
      path.moveTo(RAVELED_RES/2,RAVELED_RES/2);
      for (let idx=0; idx<resampled.length; idx++) {
        let val = 255;
        if (resampled[idx] > 0) {
          val = 0;

          let i=Math.floor(idx/256);
          let j=Math.floor(idx%256);

          if (i > j) {
            let theta1 = i * 2 * Math.PI / 256;
            let theta2 = j * 2 * Math.PI / 256;
            path.lineTo(RAVELED_RES*(0.5+Math.sin(theta1)/2.0),
                        RAVELED_RES*(0.5+Math.cos(theta1)/2.0));
            path.lineTo(RAVELED_RES*(0.5+Math.sin(theta2)/2.0),
                        RAVELED_RES*(0.5+Math.cos(theta2)/2.0));
          }
        }
        imgData.data[4*idx]   = val;
        imgData.data[4*idx+1] = val;
        imgData.data[4*idx+2] = val;
        imgData.data[4*idx+3] = 255;
      }

      raveledCtx.clearRect(0,0,RAVELED_RES,RAVELED_RES);
      raveledCtx.stroke(path);
      path = null;

      tangledCtx.putImageData(imgData, 0, 0);
      tf.engine().endScope();
    }
  }
}

window.addEventListener("DOMContentLoaded", init);