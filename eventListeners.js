trainButton.addEventListener("click", train);
testButton.addEventListener("click", test);
loadWeightsButton.addEventListener("click", loadWeights);
predictButton.addEventListener("click", predict);

clearButton.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  prediction.innerHTML = "";
});

let start = false;

if (ctx) {
  ctx.lineCap = "round";
  ctx.lineWidth = 20;
}

const initial = (e) => {
  start = true;
  if (ctx) {
    ctx.beginPath();
    ctx.moveTo(
      e.clientX - ctx.canvas.getBoundingClientRect().x,
      e.clientY - ctx.canvas.getBoundingClientRect().y
    );
  }
};

const draw = (e) => {
  if (start === true) {
    if (ctx) {
      ctx.lineTo(
        e.clientX - ctx.canvas.getBoundingClientRect().x,
        e.clientY - ctx.canvas.getBoundingClientRect().y
      );
      ctx.stroke();
      ctx.strokeStyle = "white";
    }
  }
};

canvas.addEventListener("mousedown", initial);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", () => {
  start = false;
});
