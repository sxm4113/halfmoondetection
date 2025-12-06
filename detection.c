#include <math.h>
#include <stdio.h>
#include <stdlib.h>


typedef struct {
  double x, y;
} Point;

// --- Load PGM image ---
unsigned char* loadPGM(const char* filename, int* width, int* height) {
  FILE* f = fopen(filename, "rb");
  if (!f) {
    perror("File open");
    exit(1);
  }

  char magic[3];
  fscanf(f, "%2s", magic);
  if (strcmp(magic, "P5") != 0) {
    fprintf(stderr, "Not a binary PGM (P5)\n");
    exit(1);
  }

  // Skip comments
  int c;
  do {
    c = fgetc(f);
  } while (c == '#');
  ungetc(c, f, f);

  fscanf(f, "%d %d", width, height);
  int maxval;
  fscanf(f, "%d", &maxval);
  fgetc(f);  // consume newline

  int size = (*width) * (*height);
  unsigned char* data = (unsigned char*)malloc(size);
  fread(data, 1, size, f);
  fclose(f);
  return data;
}

// --- Fit line using PCA-style covariance ---
void fitLine(Point* pts, int n, double* vx, double* vy, double* x0,
             double* y0) {
  double meanX = 0, meanY = 0;
  for (int i = 0; i < n; i++) {
    meanX += pts[i].x;
    meanY += pts[i].y;
  }
  meanX /= n;
  meanY /= n;

  double num = 0, den = 0;
  for (int i = 0; i < n; i++) {
    double dx = pts[i].x - meanX;
    double dy = pts[i].y - meanY;
    num += dx * dy;
    den += dx * dx - dy * dy;
  }

  double theta = 0.5 * atan2(2 * num, den);
  *vx = cos(theta);
  *vy = sin(theta);
  *x0 = meanX;
  *y0 = meanY;
}

// --- Projection scalar ---
double projection(Point p, double vx, double vy, double x0, double y0) {
  return (p.x - x0) * vx + (p.y - y0) * vy;
}

int main() {
  int w, h;
  unsigned char* img = loadPGM("data/halfmoon.pgm", &w, &h);

  // Collect edge pixels (simple threshold)
  Point* pts = (Point*)malloc(w * h * sizeof(Point));
  int n = 0;
  for (int y = 1; y < h - 1; y++) {
    for (int x = 1; x < w - 1; x++) {
      int idx = y * w + x;
      if (img[idx] > 127) {
        // crude edge detection: check neighbors
        if (img[idx - 1] < 127 || img[idx + 1] < 127 || img[idx - w] < 127 ||
            img[idx + w] < 127) {
          pts[n].x = x;
          pts[n].y = y;
          n++;
        }
      }
    }
  }
  printf("Collected %d edge points\n", n);

  // Fit line
  double vx, vy, x0, y0;
  fitLine(pts, n, &vx, &vy, &x0, &y0);

  // Find endpoints
  double tmin = 1e9, tmax = -1e9;
  for (int i = 0; i < n; i++) {
    double t = projection(pts[i], vx, vy, x0, y0);
    if (t < tmin) tmin = t;
    if (t > tmax) tmax = t;
  }

  Point p1 = {x0 + tmin * vx, y0 + tmin * vy};
  Point p2 = {x0 + tmax * vx, y0 + tmax * vy};
  Point center = {(p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0};

  printf("Line direction: (%.3f, %.3f)\n", vx, vy);
  printf("Midpoint on line: (%.2f, %.2f)\n", center.x, center.y);

  free(img);
  free(pts);
  return 0;
}
