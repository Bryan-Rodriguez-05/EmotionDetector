package com.example.emotiondetector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var interpreter: Interpreter
    private lateinit var emotionTextView: TextView

    // Selección de cámara para usar la cámara frontal por defecto
    private var cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        emotionTextView = findViewById(R.id.statusTextView) // Usamos el TextView existente para mostrar la emoción
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Cargar el modelo TFLite
        interpreter = Interpreter(loadModelFile())

        startCamera()
    }

    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd("model.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val modelBytes = ByteArray(fileDescriptor.declaredLength.toInt())
        inputStream.read(modelBytes)
        val buffer = ByteBuffer.allocateDirect(fileDescriptor.declaredLength.toInt())
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(modelBytes)
        return buffer
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(48, 48))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            // Procesamiento de la imagen en tiempo real
            imageAnalysis.setAnalyzer(cameraExecutor, ImageAnalysis.Analyzer { imageProxy ->
                processImage(imageProxy)
            })

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e("CameraX", "Error al inicializar la cámara", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        val buffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)

        // Convertir el buffer de bytes a un Bitmap
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        if (bitmap != null) {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 48, 48, true)
            val floatArray = preprocessImage(resizedBitmap)  // Normaliza la imagen para que sea entre 0 y 1
            val results = runModel(floatArray)

            Log.d("EmotionDetector", "Resultado del modelo: ${results.joinToString()}")
        } else {
            Log.e("EmotionDetector", "El bitmap es nulo, no se pudo procesar la imagen.")
        }


        // Cerrar la imagen para que se pueda procesar la siguiente
        imageProxy.close()
    }


    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val floatArray = FloatArray(48 * 48)
        for (y in 0 until 48) {
            for (x in 0 until 48) {
                val pixel = bitmap.getPixel(x, y)
                val gray = (0.2989 * Color.red(pixel) + 0.5870 * Color.green(pixel) + 0.1140 * Color.blue(pixel)).toFloat()
                floatArray[y * 48 + x] = gray / 255.0f // Asegúrate de normalizar entre 0 y 1
            }
        }
        return floatArray
    }


    private fun runModel(input: FloatArray): FloatArray {
        val inputBuffer = ByteBuffer.allocateDirect(4 * input.size)
        inputBuffer.order(ByteOrder.nativeOrder())
        inputBuffer.asFloatBuffer().put(input)

        val outputBuffer = ByteBuffer.allocateDirect(4 * 7)  // Número de clases (7 emociones)
        outputBuffer.order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)

        val outputArray = FloatArray(7)
        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().get(outputArray)

        return outputArray
    }


    private fun getEmotionFromOutput(output: FloatArray): String {
        val emotions = arrayOf("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: -1
        return if (maxIndex != -1) emotions[maxIndex] else "Unknown"
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        interpreter.close()
    }
}
