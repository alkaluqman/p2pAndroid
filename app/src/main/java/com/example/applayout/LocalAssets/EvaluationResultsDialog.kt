package com.example.applayout.LocalAssets

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.example.applayout.Models.evaluationApi


@Composable
fun EvaluationResultsDialog(
    isUploadable: Boolean,
    results: List<Pair<Int, Int>>,
    classLabels: List<String>,
    onDismiss: () -> Unit,
    onUpload: (evaluationApi) -> Unit
) {
    val totalPredictions = results.size
    val correctPredictions = results.count { it.first == it.second }
    val overallAccuracy = if (totalPredictions > 0) {
        correctPredictions.toDouble() / totalPredictions
    } else 0.0

    // Calculate per-class accuracy and precision
    val classCounts = classLabels.indices.associateWith { index ->
        results.count { it.second == index }
    }

    val classCorrect = classLabels.indices.associateWith { index ->
        results.count { it.first == index && it.second == index }
    }

    val classAccuracy = classLabels.indices.associateWith { index ->
        if ((classCounts[index] ?: 0) > 0) {
            (classCorrect[index] ?: 0).toDouble() / (classCounts[index] ?: 1)
        } else 0.0
    }

    val classPrecision = classLabels.indices.associateWith { index ->
        val predictedCount = results.count { it.first == index }
        if (predictedCount > 0) {
            (classCorrect[index] ?: 0).toDouble() / predictedCount
        } else 0.0
    }
    val overallPrecision = if (classPrecision.isNotEmpty()) {
        classPrecision.values.average()
    } else 0.0

    val classPerformance = classLabels.indices.map { index ->
        classAccuracy[index] ?: 0.0
    }
    val evaluationDate = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        .format(java.util.Date())
    Log.d("eval", classCounts.toString())
    Log.d("eval", classCorrect.toString())
    Log.d("eval", classAccuracy.toString())
    Log.d("eval", classPrecision.toString())
    Log.d("eval", overallPrecision.toString())
    Log.d("eval", classPerformance.toString())

    Dialog(onDismissRequest = onDismiss) {
        Box(
            modifier = Modifier
                .background(MaterialTheme.colorScheme.surface)
                .padding(16.dp)
        ) {
            LazyColumn(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Add the header
                item {
                    Text("Evaluation Results", style = MaterialTheme.typography.titleLarge)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        "Overall Accuracy: ${(overallAccuracy * 100).format(2)}%",
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                }

                // Add the list of class items
                items(classLabels.indices.toList()) { index ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("Class ${index} (${classLabels[index]}):")
                        Text(
                            "Accuracy: ${
                                (classAccuracy[index]?.times(100)?.format(2)) ?: "0.00"
                            }%"
                        )
                        Text(
                            "Precision: ${
                                (classPrecision[index]?.times(100)?.format(2)) ?: "0.00"
                            }%"
                        )
                    }
                }

                // Add the footer (buttons)
                item {
                    Spacer(modifier = Modifier.height(16.dp))
                    Button(
                        onClick = {
                            onUpload(
                                evaluationApi(
                                    accuracy = overallAccuracy,
                                    precision = overallPrecision,
                                    class_performance = classPerformance,
                                    evaluationDate = evaluationDate
                                )
                            )
                        },
                        enabled = isUploadable
                    ) {
                        Text("Upload Results")
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(onClick = onDismiss) {
                        Text("Close")
                    }
                }
            }
        }
    }

}

fun Double.format(digits: Int) = "%.${digits}f".format(this)