package com.example.applayout.Assets.Datasets

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.applayout.Database.Entities.Dataset
import java.util.UUID

@Composable
fun AddDatasetDialog(
    onDismiss: () -> Unit,
    onAddDataset: (Dataset) -> Unit
) {
    var datasetName by remember { mutableStateOf("") }
    var datasetDescription by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = {
                if (datasetName.isNotBlank()) {
                    val datasetData = Dataset(
                        datasetId = UUID.randomUUID().toString(),
                        name = datasetName,
                        description = datasetDescription,
                    )
                    onAddDataset(datasetData)
                    onDismiss()
                }
            }) {
                Text("Add")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
        title = { Text("Add New Model") },
        text = {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text("Model name:")
                BasicTextField(
                    value = datasetName,
                    onValueChange = { datasetName = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp)
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text("Description:")
                BasicTextField(
                    value = datasetDescription,
                    onValueChange = { datasetDescription = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp)
                )

            }
        }
    )
}