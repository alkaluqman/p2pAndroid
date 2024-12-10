package com.example.applayout.Database.Daos

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.applayout.Database.Entities.Dataset
import kotlinx.coroutines.flow.Flow

@Dao
interface DatasetDao {
    @Query("DELETE FROM dataset WHERE datasetId = :datasetId")
    fun deleteDatasetById(datasetId: String)

    @Query("SELECT * FROM dataset WHERE datasetId=:datasetId")
    fun getDataset(datasetId: String): LiveData<Dataset>

    @Query("SELECT * FROM dataset")
    fun getAllDataset(): Flow<List<Dataset>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertDataset(dataset: Dataset)

    @Query("UPDATE dataset SET name = :name, description = :description, dirPath = :dirPath, lastModified = :lastModified WHERE datasetId = :id")
    fun updateDatasetFields(
        id: String,
        name: String,
        dirPath: String,
        lastModified: Long = System.currentTimeMillis(),
        description: String,
    )

}