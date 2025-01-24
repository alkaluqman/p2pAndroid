package com.example.applayout.Data.Database.Daos


import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.example.applayout.Data.Model.LocalModel


@Dao
interface LocalModelDao {
    @Query("SELECT * FROM LocalModels WHERE uniqueIdentifier=:modelId")
    fun getModel(modelId: String): LocalModel?

    @Query("SELECT * FROM LocalModels")
    fun getAllModels(): List<LocalModel>

    @Query("UPDATE LocalModels SET description = :description, model_task = :task WHERE uniqueIdentifier = :uniqueIdentifier")
    fun updateModel(
        uniqueIdentifier: String,
        description: String,
        task: String,
    )

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertModel(model: LocalModel)

    @Query("DELETE FROM LocalModels WHERE uniqueIdentifier = :uniqueIdentifier")
    fun deleteModel(uniqueIdentifier: String)
}