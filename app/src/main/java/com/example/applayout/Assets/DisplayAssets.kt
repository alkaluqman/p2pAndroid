package com.example.applayout.Assets

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.applayout.Assets.Datasets.AddDatasetDialog
import com.example.applayout.Assets.Datasets.DatasetCard
import com.example.applayout.Assets.Models.AddModelDialog
import com.example.applayout.Assets.Models.ModelCard
import java.io.File


@Composable
fun DisplayAssets(filesDir: File, viewModel: AssetsViewModel) {
    var showModelDialog by remember { mutableStateOf(false) }
    var showDatasetDialog by remember { mutableStateOf(false) }
    val modelList by viewModel.getModelsNotDeleted().collectAsState(initial = emptyList())
    val datasetList by viewModel.getDataset().collectAsState(initial = emptyList())

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        item {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Datasets",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
                Button(
                    onClick = { showDatasetDialog = true }, //TODO for dataset
                    modifier = Modifier.align(Alignment.CenterVertically)
                ) {
                    Text("Add Dataset")
                }
            }
        }
        items(datasetList) { dataset ->
            DatasetCard(filesDir = filesDir, datasetId = dataset.datasetId)
        }
        item {
            Spacer(modifier = Modifier.height(16.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Models",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
                Button(
                    onClick = { showModelDialog = true },
                    modifier = Modifier.align(Alignment.CenterVertically)
                ) {
                    Text("Add Model")
                }
            }
        }

        items(modelList) { model ->
            ModelCard(filesDir = filesDir, modelId = model.modelId)
        }
    }
    if (showModelDialog) {
        AddModelDialog(
            onDismiss = { showModelDialog = false },
            onAddModel = { modelData ->
                viewModel.addModel(modelData, filesDir)
                showModelDialog = false
            }
        )
    }
    if (showDatasetDialog) {
        AddDatasetDialog(
            onDismiss = { showDatasetDialog = false },
            onAddDataset = { modelData ->
                viewModel.addDataset(modelData, filesDir)
                showModelDialog = false
            }
        )
    }
}



