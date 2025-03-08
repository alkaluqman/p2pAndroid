package com.example.applayout.Data.Model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "LocalModels")
data class LocalModel(
    @PrimaryKey val uniqueIdentifier: String,
    val model_task: String = "object_detection",
    val description: String = "",
    val absoluteFilePath: String = ""
)