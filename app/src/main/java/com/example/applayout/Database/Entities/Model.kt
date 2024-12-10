package com.example.applayout.Database.Entities

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.TypeConverters
import com.example.applayout.Models.SyncStatus
import com.example.applayout.Models.SyncStatusConverter

@Entity(tableName = "models")
data class Model(
    @PrimaryKey val modelId: String,
    var name: String = "",
    var filePath: String = "",
    @TypeConverters(SyncStatusConverter::class) var syncStatus: SyncStatus = SyncStatus.PENDING,
    var lastModified: Long = System.currentTimeMillis(),
    var description: String = "",
    var architecture: String = "",
    var likes: Int = 0,
    var usage: Int = 0,
    var task: String = ""
)