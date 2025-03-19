import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
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
        } else null
    }

    val validPrecisions = classPrecision.values.filterNotNull()
    val overallPrecision = if (validPrecisions.isNotEmpty()) {
        validPrecisions.average()
    } else 0.0

    val evaluationDate = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        .format(java.util.Date())

    Dialog(onDismissRequest = onDismiss) {
        Surface(
            shape = MaterialTheme.shapes.medium,
            tonalElevation = 8.dp,
            modifier = Modifier.padding(16.dp)
        ) {
            Column(
                modifier = Modifier
                    .padding(16.dp)
                    .fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Evaluation Results",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                Divider(modifier = Modifier.fillMaxWidth())
                Spacer(modifier = Modifier.height(16.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Accuracy",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "${(overallAccuracy * 100).format(2)}%",
                            style = MaterialTheme.typography.bodyLarge
                        )
                    }
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Precision",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "${(overallPrecision * 100).format(2)}%",
                            style = MaterialTheme.typography.bodyLarge
                        )
                    }
                }
                Divider(modifier = Modifier.fillMaxWidth())
                Spacer(modifier = Modifier.height(16.dp))

                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    items(classLabels.indices.toList()) { index ->
                        Surface(
                            shape = MaterialTheme.shapes.medium,
                            tonalElevation = 2.dp,
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 4.dp, horizontal = 8.dp)
                        ) {
                            Row(
                                modifier = Modifier
                                    .padding(8.dp)
                                    .fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = classLabels[index],
                                    style = MaterialTheme.typography.bodyMedium,
                                    modifier = Modifier.weight(2f)
                                )
                                Text(
                                    text = "Correct: ${classCorrect[index] ?: 0} / ${classCounts[index] ?: 0}",
                                    style = MaterialTheme.typography.bodySmall,
                                    modifier = Modifier.weight(2f)
                                )
                            }
                        }
                    }
                }
                Row(modifier = Modifier.fillMaxWidth()) {
                    Spacer(modifier = Modifier.weight(1f))
                    Text(
                        text = "Evaluation Date: $evaluationDate",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))
                // Buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Button(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Close")
                    }
                    Spacer(modifier = Modifier.width(16.dp))
                    Button(
                        onClick = {
                            onUpload(
                                evaluationApi(
                                    accuracy = overallAccuracy,
                                    precision = overallPrecision,
                                    class_performance = classLabels.indices.map {
                                        classAccuracy[it] ?: 0.0
                                    },
                                    evaluationDate = evaluationDate
                                )
                            )
                        },
                        enabled = isUploadable,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Upload")
                    }
                }
            }
        }
    }
}

fun Double.format(digits: Int) = "%.${digits}f".format(this)
