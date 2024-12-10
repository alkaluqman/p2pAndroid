package com.example.applayout.Assets.Datasets

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.example.applayout.Database.AppDatabase
import com.example.applayout.Database.Entities.Dataset
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class DatasetDetailsActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val datasetId = intent.getStringExtra("datasetId")
        if (datasetId == null) {
            Toast.makeText(this, "Invalid dataset data", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val database = AppDatabase.getDatabase(applicationContext)

        setContent {
            MaterialTheme {
                val datasetData by database.datasetDao().getDataset(datasetId).observeAsState()
                datasetData?.let { dataset ->
                    DatasetDetailsScreen(
                        datasetData = dataset,
                        onSave = { updatedDatasetData ->
                            lifecycleScope.launch(Dispatchers.IO) {
                                updatedDatasetData.lastModified = System.currentTimeMillis()
                                database.datasetDao().updateDatasetFields(
                                    updatedDatasetData.datasetId,
                                    updatedDatasetData.name,
                                    updatedDatasetData.dirPath,
                                    updatedDatasetData.lastModified,
                                    updatedDatasetData.description
                                )
                            }
                            Toast.makeText(
                                this@DatasetDetailsActivity,
                                "Metadata saved!",
                                Toast.LENGTH_SHORT
                            ).show()
                            finish()
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun DatasetDetailsScreen(datasetData: Dataset, onSave: (Dataset) -> Unit) {
    var datasetName by remember { mutableStateOf(datasetData.name) }
    var description by remember { mutableStateOf(datasetData.description) }


    Column(modifier = Modifier
        .padding(16.dp)
        .fillMaxSize()) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.primary)
                .padding(16.dp)
        ) {
            Text(
                text = "Edit Model Details",
                color = MaterialTheme.colorScheme.onPrimary,
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.align(Alignment.Center)
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text("Dataset Name:")
        BasicTextField(
            value = datasetName,
            onValueChange = { datasetName = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        )

        Text("Description:")
        BasicTextField(
            value = description,
            onValueChange = { description = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        )


        Spacer(modifier = Modifier.height(16.dp))
        Button(onClick = {
            onSave(
                Dataset(
                    datasetId = datasetData.datasetId,
                    name = datasetName,
                    description = description,
                    lastModified = System.currentTimeMillis()
                )
            )
        }) {
            Text("Save Changes")
        }
    }
}
