package com.example.applayout.Dataset

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Create
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.applayout.Data.Model.Dataset
import com.example.applayout.Models.InfoText

@Composable
fun DatasetCard(
    datasetData: Dataset,
    numImages: Int,
    onRemove: (String) -> Unit,
    onEdit: (Dataset) -> Unit,
    onRun: (Dataset) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = datasetData.uniqueIdentifier,
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp,
                    modifier = Modifier.padding(bottom = 4.dp),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column() {
                        Column(modifier = Modifier.weight(2f)) {
                            InfoText("Number of Images:", numImages.toString())
                        }
                        Column(modifier = Modifier.weight(2f)) {
                            InfoText("Description:", datasetData.description)
                        }
                    }

                    Row {
                        IconButton(onClick = { onEdit(datasetData) }) {
                            Icon(
                                imageVector = Icons.Default.Create,
                                contentDescription = "Edit Dataset",
                            )
                        }
                        IconButton(onClick = { onRemove(datasetData.uniqueIdentifier) }) {
                            Icon(
                                imageVector = Icons.Default.Delete,
                                contentDescription = "Delete Dataset",
                            )
                        }
                        IconButton(onClick = { onRun(datasetData) }) {
                            Icon(
                                imageVector = Icons.Default.PlayArrow,
                                contentDescription = "Run Inference",
                            )
                        }
                    }

                }

            }
        }
    }
}


