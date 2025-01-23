package com.example.applayout.Data.Database.Daos

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.applayout.Data.Model.Dataset

@Dao
interface LocalDatasetDao {
    @Query("SELECT * FROM LocalDatasets WHERE uniqueIdentifier=:datasetId")
    fun getDataset(datasetId: String): Dataset?

    @Query("SELECT * FROM LocalDatasets")
    fun getAllDatasets(): List<Dataset>

    @Query("UPDATE LocalDatasets SET description = :description, model_task = :task WHERE uniqueIdentifier = :uniqueIdentifier")
    fun updateDataset(
        uniqueIdentifier: String,
        description: String,
        task: String,
    )

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertDataset(dataset: Dataset)

    @Query("DELETE FROM LocalDatasets WHERE uniqueIdentifier = :uniqueIdentifier")
    fun deleteDataset(uniqueIdentifier: String)
}