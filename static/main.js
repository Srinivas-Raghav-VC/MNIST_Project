var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
var mouse = { x: 0, y: 0 };

canvas.addEventListener("mousemove", function (e) {
  mouse.x = e.pageX - this.offsetLeft;
  mouse.y = e.pageY - this.offsetTop;
});

canvas.onmousedown = () => {
  ctx.beginPath();
  ctx.moveTo(mouse.x, mouse.y);
  canvas.addEventListener("mousemove", onPaint);
};

canvas.onmouseup = () => {
  canvas.removeEventListener("mousemove", onPaint);
};

var onPaint = () => {
  ctx.lineWidth = 10;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = "#000000";
  ctx.lineTo(mouse.x, mouse.y);
  ctx.stroke();
};

document
  .getElementById("predict-button")
  .addEventListener("click", function () {
    var dataURL = canvas.toDataURL("image/png");
    $.ajax({
      type: "POST",
      url: "/predict/",
      data: JSON.stringify({
        image: dataURL.split(",")[1],
      }),
      contentType: "application/json",
      dataType: "json",
      success: function (o) {
        console.log("predicted digit: " + o);
        alert("Predicted digit: " + o);
      },
    });
  });
