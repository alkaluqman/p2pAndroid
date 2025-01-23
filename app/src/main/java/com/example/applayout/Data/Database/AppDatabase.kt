package com.example.applayout.Data.Database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.example.applayout.Data.Database.Daos.LocalDatasetDao
import com.example.applayout.Data.Database.Daos.LocalModelDao
import com.example.applayout.Data.Database.Daos.LocalRelationshipDao
import com.example.applayout.Data.Model.Dataset
import com.example.applayout.Data.Model.LocalModel
import com.example.applayout.Data.Model.LocalRelationship

@Database(entities = [LocalModel::class, LocalRelationship::class, Dataset::class], version = 1)
abstract class AppDatabase : RoomDatabase() {
    abstract fun localModelDao(): LocalModelDao
    abstract fun localRelationshipDao(): LocalRelationshipDao
    abstract fun localDatasetDao(): LocalDatasetDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "room_database"
                )
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}