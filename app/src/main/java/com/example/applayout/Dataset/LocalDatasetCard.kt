package com.example.applayout.Dataset

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Create
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun LocalDatasetCard(
    datasetName: String,
    onUpload: (String) -> Unit,
    onRemove: (String) -> Unit,
    onEdit: (String) -> Unit
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
                    text = datasetName,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                )
                Row() {
                    IconButton(onClick = { onEdit(datasetName) }) {
                        Icon(
                            imageVector = Icons.Default.Create,
                            contentDescription = "Edit Dataset",
                        )
                    }
                    IconButton(onClick = { onUpload(datasetName) }) {
                        Icon(
                            imageVector = Icons.Default.Send,
                            contentDescription = "Upload Dataset",
                        )
                    }
                    IconButton(onClick = { onRemove(datasetName) }) {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = "Delete Dataset",
                        )
                    }
                }
            }
        }
    }
}


