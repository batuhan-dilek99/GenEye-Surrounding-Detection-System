/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Log;
import android.util.Pair;
import android.util.TypedValue;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;
import android.speech.tts.TextToSpeech;
import java.util.Locale;
import org.tensorflow.lite.examples.detection.MainActivity;


/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
          Color.BLUE,
          Color.RED,
          Color.GREEN,
          Color.YELLOW,
          Color.CYAN,
          Color.MAGENTA,
          Color.WHITE,
          Color.parseColor("#55FF55"),
          Color.parseColor("#FFA500"),
          Color.parseColor("#FF8888"),
          Color.parseColor("#AAAAFF"),
          Color.parseColor("#FFFFAA"),
          Color.parseColor("#55AAAA"),
          Color.parseColor("#AA33AA"),
          Color.parseColor("#0D0068")
  };
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint boxPaint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;

  private TextToSpeech textToSpeech;

  private int mode;
  private int canvaswidth;
  private int canvasheight;
  private int flagThreat;
  private int flagThreatLeft;
  private int flagThreatRight;
  private int flagRight;
  private int flagLeft;
  private int flagFront;
  private int flagThread;
  private boolean objLeft;
  private boolean objRight;
  public MultiBoxTracker(final Context context, TextToSpeech textToSpeech, int mode) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }
    this.textToSpeech = textToSpeech;
    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(10.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);
    this.mode = mode;
    flagThread = 0;
    textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
          final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public void guidance(int mode){
    if(mode == 1){
      if (objRight){
        textToSpeech.speak("it is little bit to right", TextToSpeech.QUEUE_FLUSH, null);
      }
      else if (objLeft){
        textToSpeech.speak("it is little bit to left", TextToSpeech.QUEUE_FLUSH, null);
      }
      else {
        textToSpeech.speak("it is in front of you", TextToSpeech.QUEUE_FLUSH, null);
      }
    }
    else if (mode == 2){
      if (flagThreat == 1){
        if (flagFront == 1){
          textToSpeech.speak("threat in front of you", TextToSpeech.QUEUE_FLUSH, null);
        }
        if (flagRight == 1 && flagThreatRight == 1){
          textToSpeech.speak("go left", TextToSpeech.QUEUE_FLUSH, null);
        }
        if (flagLeft == 1 && flagThreatLeft == 1){
          textToSpeech.speak("go right", TextToSpeech.QUEUE_FLUSH, null);
        }
      }
    }
  }

  class OutputThread extends Thread{
    int milSeconds;
    @Override
    public void run(){
      if (mode == 1){
        milSeconds = 2200;
      }
      else if (mode == 2){
        milSeconds = 1500;
      }
      if (flagThread == 0) {
        flagThread = 1;

        guidance(mode);
        try {
          Thread.sleep(milSeconds);
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
        flagThread = 0;
      }

    }
  }

  public synchronized void draw(final Canvas canvas) {
    canvaswidth = canvas.getWidth();
    canvasheight = canvas.getHeight();
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
            Math.min(
                    canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                    canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
            ImageUtils.getTransformationMatrix(
                    frameWidth,
                    frameHeight,
                    (int) (multiplier * (rotated ? frameHeight : frameWidth)),
                    (int) (multiplier * (rotated ? frameWidth : frameHeight)),
                    sensorOrientation,
                    false);
    for (final TrackedRecognition recognition : trackedObjects) {
      final RectF trackedPos = new RectF(recognition.location);

      getFrameToCanvasMatrix().mapRect(trackedPos);



      int upperBoundary = ((canvaswidth * 45) / 100 + canvaswidth / 7);
      int lowerBoundary = ((canvaswidth * 45) / 100 - canvaswidth / 7);
      if (mode == 1){
        if (trackedPos.centerX() > ((canvaswidth * 45) / 100 + canvaswidth / 7)){
          objLeft = false;
          objRight = true;
          OutputThread thread = new OutputThread();
          thread.start();
        }
        else if (trackedPos.centerX() < ((canvaswidth * 45) / 100 - canvaswidth / 7)){
          objRight = false;
          objLeft = true;
          OutputThread thread = new OutputThread();
          thread.start();
        }
        else {
          objRight = false;
          objLeft = false;
          OutputThread thread = new OutputThread();
          thread.start();
        }
      }
      else if (mode == 2){
        if (trackedPos.top < (canvasheight * 60) / 100){
          flagThreat = 1;
          if (trackedPos.centerX() < (canvaswidth * 45) / 100){
            flagLeft = 1;
            flagRight = 0;
            if (trackedPos.right > lowerBoundary){
              flagThreatLeft = 1;
              OutputThread thread = new OutputThread();
              thread.start();
            }
            else{
              flagThreatLeft = 0;
            }
          }
          else if(trackedPos.centerX() > (canvaswidth * 45) / 100){
            flagRight = 1;
            flagLeft = 0;
            if (trackedPos.left < upperBoundary){
              flagThreatRight = 1;
              OutputThread thread = new OutputThread();
              thread.start();
            }
            else {
              flagThreatRight = 0;
            }
          }
          if (trackedPos.right > lowerBoundary && trackedPos.left < upperBoundary){
            flagFront = 1;
          }
          else{
            flagFront = 0;
          }
        }
        else {
          flagThreat = 0;
        }
      }

      boxPaint.setColor(recognition.color);
      float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
      canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

      final String labelString =
              !TextUtils.isEmpty(recognition.title)
                      ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
                      : String.format("%.2f", (100 * recognition.detectionConfidence));
      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
      // labelString);
      borderedText.drawText(
              canvas, trackedPos.left + cornerSize, trackedPos.top, labelString + "%", boxPaint);
    }
  }


  private void processResults(final List<Recognition> results) {
    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());
    boolean voice_output = true;
    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());
      final RectF detectionScreenRect = new RectF();


      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
              "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }

    trackedObjects.clear();
    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    for (final Pair<Float, Recognition> potential : rectsToTrack) {
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      trackedRecognition.detectionConfidence = potential.first;
      trackedRecognition.location = new RectF(potential.second.getLocation());
      trackedRecognition.title = potential.second.getTitle();
//      trackedRecognition.color = COLORS[trackedObjects.size() % COLORS.length];
      trackedRecognition.color = COLORS[potential.second.getDetectedClass() % COLORS.length];
      trackedObjects.add(trackedRecognition);

//      if (trackedObjects.size() >= COLORS.length) {
//        break;
//      }
    }
  }

  private static class TrackedRecognition {
    RectF location;
    float detectionConfidence;
    int color;
    String title;
  }
}
