package com.example.applayout.Models

import androidx.room.TypeConverter

enum class SyncStatus {
    SYNCED,
    PENDING,
    DELETED
}

class SyncStatusConverter {
    @TypeConverter
    fun fromSyncStatus(status: SyncStatus): String {
        return status.name // Convert enum to String
    }

    @TypeConverter
    fun toSyncStatus(value: String): SyncStatus {
        return SyncStatus.valueOf(value.uppercase())
    }
}