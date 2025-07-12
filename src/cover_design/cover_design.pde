int numSpores = 50;

PVector[] sporeCenters;

float diffRad;
float x, y, z;

// image size
int cWidth = 5000;
int cHeight = 4000;
float prevScale = 0.25;
int prevWidth = int(cWidth * prevScale);
int prevHeight = int(cHeight * prevScale);

PGraphics canvas;

void settings(){
  size(prevWidth, prevHeight, P3D);
}

void setup(){
  
  //size(1000, 1000, P3D);
  
  diffRad = 1000;
  
  sporeCenters = new PVector[numSpores];
  
  canvas = createGraphics(cWidth, cHeight, P3D);
  canvas.beginDraw();
  canvas.background(240, 245, 254);
  //canvas.background(0);
  canvas.strokeWeight(5);
  canvas.endDraw();
  
  for(int i = 0; i < numSpores; i++){
    sporeCenters[i] = new PVector(random(-cWidth, cWidth), random(-cWidth, cWidth), random(-cWidth, cWidth));
  }
  
  
  
  
  //camera(70.0, 35.0, 120.0, 50.0, 50.0, 0.0, 0.0, 1.0, 0.0);
}

void draw(){
  
  canvas.beginDraw();
  
  for(PVector sporeCen : sporeCenters){
    x = randomGaussian() * diffRad + sporeCen.x;
    y = randomGaussian() * diffRad + sporeCen.y;
    z = randomGaussian() * diffRad + sporeCen.z;
    
    canvas.fill(220, 220, (cos(sporeCen.x*sporeCen.y*0.01)*1+1)*110, 1);
    
    canvas.noStroke();
    canvas.pushMatrix();
    canvas.translate(sporeCen.x, sporeCen.y, sporeCen.z);
    canvas.sphere(50);
    canvas.popMatrix();
    
    canvas.stroke(230, 230, 0, 100);
    canvas.point(x, y, z);
  }
  
  canvas.endDraw();
  image(canvas, 0, 0, prevWidth, prevHeight);
}

void keyPressed(){
  if (key == ' ') {
    canvas.save("output.tiff");
  }
}
