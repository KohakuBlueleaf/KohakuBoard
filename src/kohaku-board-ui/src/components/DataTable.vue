<script setup>
import { ref, computed } from "vue";

const props = defineProps({
  data: {
    type: Object,
    required: true,
    // Format: { columns: ['col1', 'col2'], rows: [[val1, val2], ...] }
  },
  title: {
    type: String,
    default: "",
  },
  maxHeight: {
    type: String,
    default: "400px",
  },
});

const sortColumn = ref(null);
const sortDirection = ref("asc");

const sortedRows = computed(() => {
  if (!sortColumn.value) return props.data.rows;

  const columnIndex = props.data.columns.indexOf(sortColumn.value);
  const sorted = [...props.data.rows].sort((a, b) => {
    const aVal = a[columnIndex];
    const bVal = b[columnIndex];

    if (typeof aVal === "number" && typeof bVal === "number") {
      return sortDirection.value === "asc" ? aVal - bVal : bVal - aVal;
    }

    const aStr = String(aVal);
    const bStr = String(bVal);
    return sortDirection.value === "asc"
      ? aStr.localeCompare(bStr)
      : bStr.localeCompare(aStr);
  });

  return sorted;
});

function toggleSort(column) {
  if (sortColumn.value === column) {
    sortDirection.value = sortDirection.value === "asc" ? "desc" : "asc";
  } else {
    sortColumn.value = column;
    sortDirection.value = "asc";
  }
}
</script>

<template>
  <div class="w-full">
    <h3 v-if="title" class="text-lg font-semibold mb-3">{{ title }}</h3>
    <div
      class="overflow-auto border border-gray-200 dark:border-gray-700 rounded-lg"
      :style="{ maxHeight }"
    >
      <table class="w-full text-sm">
        <thead
          class="sticky top-0 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700"
        >
          <tr>
            <th
              v-for="column in data.columns"
              :key="column"
              @click="toggleSort(column)"
              class="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300 cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              <div class="flex items-center gap-2">
                {{ column }}
                <span v-if="sortColumn === column" class="text-xs">
                  {{ sortDirection === "asc" ? "↑" : "↓" }}
                </span>
              </div>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(row, index) in sortedRows"
            :key="index"
            class="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800"
          >
            <td
              v-for="(cell, cellIndex) in row"
              :key="cellIndex"
              class="px-4 py-3 text-gray-900 dark:text-gray-100"
            >
              {{ cell }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
