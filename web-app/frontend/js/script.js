const startDrawing = () => { 
    const button = document.querySelector("button");
    const video = document.querySelector("#videoPlayer");
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const fpsInfo = document.querySelector("#fps-info");
    const metadataInfo =  document.querySelector("#metadata-info");
    const videouploader = document.querySelector("#videoUploader");
    
    button.addEventListener('click', () => video.paused ? video.play() : video.pause());

    videouploader.addEventListener('change', function(event) {
        var file = event.target.files[0];
        var url = URL.createObjectURL(file);
        video.src = url;
        video.requestVideoFrameCallback(updateCanvas); 
    });
  
    video.addEventListener('play', () => {
      if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) {
        return alert('Your browser does not support the `Video.requestVideoFrameCallback()` API.');
      }    
    });
    
    let width = canvas.width;
    let height = canvas.height;
    
    let paintCount = 0;
    let startTime = 0.0;
  
    const updateCanvas = (now, metadata) => {
      if (startTime === 0.0) {
        startTime = now;
      }
  
      ctx.drawImage(video, 0, 0, width, height);
  
      const elapsed = (now - startTime) / 1000.0;
      const fps = (++paintCount / elapsed).toFixed(3);
      fpsInfo.innerText = !isFinite(fps) ? 0 : fps;
      metadataInfo.innerText = JSON.stringify(metadata, null, 2);
  
      video.requestVideoFrameCallback(updateCanvas);
    };
  };
  
  window.addEventListener('load', startDrawing);

