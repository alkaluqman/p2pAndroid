package com.example.applayout.Assets


import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.applayout.Database.AppDatabase
import com.example.applayout.Database.Entities.Dataset
import com.example.applayout.Database.Entities.Model
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import java.io.File

class AssetsViewModel(private val database: AppDatabase) : ViewModel() {
    fun getModelsNotDeleted(): Flow<List<Model>> {
        return database.modelDao().getModelsNotDeleted()
    }

    fun getDataset(): Flow<List<Dataset>> {
        return database.datasetDao().getAllDataset()
    }

    fun addModel(modelData: Model, filesDir: File) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val relativeFilePath = "models/${modelData.modelId}/model.tflite"
                modelData.filePath = relativeFilePath
                downloadFile(
                    filesDir,
                    relativeFilePath,
                    "https://drive.google.com/uc?export=download&id=1UFOLNPU0SO8aUdt9ydjeSdCvqZ_ATl6k"//dummy data
                )
                database.modelDao().insertModel(modelData)
                logDatabaseContents(database)
            } catch (e: Exception) {
                Log.e("ModelSaving", "Error saving model: ${e.message}", e)
            }
        }
    }

    fun addDataset(datasetData: Dataset, filesDir: File) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val relativeDirPath = "dataset/${datasetData.datasetId}"
                datasetData.dirPath = relativeDirPath
                downloadAndUnzipFile(
                    filesDir,
                    relativeDirPath,
                    "https://drive.google.com/uc?export=download&id=14hUtr9OpO8j_pzJmamsRSm5RvR6GB1PD" //dummy data
                )
                database.datasetDao().insertDataset(datasetData)
                logDatabaseContents(database)
            } catch (e: Exception) {
                Log.e("ModelSaving", "Error saving model: ${e.message}", e)
            }
        }
    }


}