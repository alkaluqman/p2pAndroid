package com.example.applayout.Database.Daos

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.applayout.Database.Entities.Model
import com.example.applayout.Models.SyncStatus
import kotlinx.coroutines.flow.Flow

@Dao
interface ModelDao {
    @Query("SELECT * FROM models WHERE modelId=:modelId")
    fun getModel(modelId: String): LiveData<Model>

    @Query("SELECT * FROM models")
    fun getAllModels(): List<Model>

    @Query("SELECT * FROM models WHERE syncStatus = :status")
    fun getModelsBySyncStatus(status: SyncStatus): List<Model>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertModel(model: Model)

    @Query("UPDATE models SET syncStatus = :syncStatus WHERE modelId = :modelId")
    fun deleteModel(modelId: String, syncStatus: SyncStatus = SyncStatus.DELETED)


    @Query("UPDATE models SET name = :name, description = :description, architecture = :architecture, task = :task, lastModified = :lastModified, syncStatus= :syncStatus WHERE modelId = :id")
    fun updateModelFields(
        id: String,
        name: String,
        description: String,
        architecture: String,
        task: String,
        lastModified: Long = System.currentTimeMillis(),
        syncStatus: SyncStatus = SyncStatus.PENDING
    )

    @Query("SELECT * FROM models WHERE syncStatus != :status")
    fun getModelsNotDeleted(status: SyncStatus = SyncStatus.DELETED): Flow<List<Model>>
}