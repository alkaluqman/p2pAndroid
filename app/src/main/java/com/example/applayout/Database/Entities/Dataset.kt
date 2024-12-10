package com.example.applayout.Database.Entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "dataset")
data class Dataset(
    @PrimaryKey val datasetId: String,
    var name: String = "",
    var dirPath: String = "",
    var lastModified: Long = System.currentTimeMillis(),
    var description: String = ""
)
