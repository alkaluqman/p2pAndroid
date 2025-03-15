package com.example.applayout.Dataset

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
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
        Column(modifier = Modifier.padding(16.dp)) {
            // Dataset Name at the Top
            Text(
                text = "Dataset: ${datasetData.uniqueIdentifier.take(8)}",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    InfoText("Number of Images:", numImages.toString())
                    InfoText("Description:", datasetData.description)
                }
                IconButton(onClick = { onEdit(datasetData) }) {
                    Icon(imageVector = Icons.Default.Create, contentDescription = "Edit Dataset")
                }
                IconButton(onClick = { onRemove(datasetData.uniqueIdentifier) }) {
                    Icon(imageVector = Icons.Default.Delete, contentDescription = "Delete Dataset")
                }
                IconButton(onClick = { onRun(datasetData) }) {
                    Icon(
                        imageVector = Icons.Default.PlayArrow,
                        contentDescription = "Run Inference"
                    )
                }
            }


        }
    }
}
