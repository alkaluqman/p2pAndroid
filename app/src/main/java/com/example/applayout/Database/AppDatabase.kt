package com.example.applayout.Database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.example.applayout.Database.Daos.DatasetDao
import com.example.applayout.Database.Daos.ModelDao
import com.example.applayout.Database.Entities.Dataset
import com.example.applayout.Database.Entities.Model
import com.example.applayout.Models.SyncStatusConverter

@Database(entities = [Dataset::class, Model::class], version = 1)
@TypeConverters(SyncStatusConverter::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun modelDao(): ModelDao
    abstract fun datasetDao(): DatasetDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "room_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}