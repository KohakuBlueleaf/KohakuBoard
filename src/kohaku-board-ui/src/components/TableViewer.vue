<script setup>
import { useSliderSync } from "@/composables/useSliderSync";

const props = defineProps({
  tableData: Array,
  height: Number,
  currentStep: Number,
  cardId: String,
  autoAdvanceToLatest: {
    type: Boolean,
    default: true,
  },
});

const emit = defineEmits(["update:currentStep", "update:autoAdvance"]);

const stepIndex = computed({
  get() {
    if (!props.tableData || props.tableData.length === 0) return 0;
    const index = props.tableData.findIndex(
      (item) => item.step === props.currentStep,
    );
    // If not found and autoAdvance enabled, return latest
    if (index === -1 && props.autoAdvanceToLatest) {
      return props.tableData.length - 1;
    }
    return index >= 0 ? index : 0;
  },
  set(index) {
    if (props.tableData && props.tableData[index]) {
      const newStep = props.tableData[index].step;
      emit("update:currentStep", newStep);

      // Check if user moved to latest - enable auto-advance
      if (index === props.tableData.length - 1) {
        if (!props.autoAdvanceToLatest) {
          emit("update:autoAdvance", true);
        }
      } else {
        // User moved away from latest - disable auto-advance
        if (props.autoAdvanceToLatest) {
          emit("update:autoAdvance", false);
        }
      }

      // Trigger synchronization if shift is pressed
      triggerSync(newStep);
    }
  },
});

// Watch for new data and auto-advance if enabled
watch(
  () => props.tableData?.length,
  (newLength, oldLength) => {
    if (props.autoAdvanceToLatest && newLength > oldLength && newLength > 0) {
      // New data and auto-advance enabled - move to latest
      const latestStep = props.tableData[newLength - 1].step;
      emit("update:currentStep", latestStep);
    }
  },
);

// Setup slider synchronization
const { isShiftPressed, triggerSync } = useSliderSync(
  computed(() => `table-${props.cardId}`),
  computed(() => props.tableData || []),
  (newStep) => {
    emit("update:currentStep", newStep);
  },
);

const experimentId = computed(() => {
  // Extract experiment ID from parent route
  const route = useRoute();
  return route.params.id;
});

const currentTable = computed(() => {
  if (!props.tableData || props.tableData.length === 0) return null;
  const index = stepIndex.value >= 0 ? stepIndex.value : 0;
  return props.tableData[index];
});

const processedTableData = computed(() => {
  if (!currentTable.value) return null;

  // Process table and convert <media id=xxx> to actual URLs
  const tableName = currentTable.value.name.replace("/", "__");
  const step = currentTable.value.step;

  console.log("Processing table:", tableName, "step:", step);
  console.log("Columns:", currentTable.value.columns);
  console.log("Column types:", currentTable.value.column_types);
  console.log("Row count:", currentTable.value.rows.length);
  console.log("First row:", currentTable.value.rows[0]);

  const processedRows = currentTable.value.rows.map((row, rowIdx) => {
    return row.map((cell, colIdx) => {
      const colType = currentTable.value.column_types[colIdx];
      console.log(
        `Row ${rowIdx} Col ${colIdx} type="${colType}" cell:`,
        typeof cell,
        cell.substring ? cell.substring(0, 50) : cell,
      );

      // Check if this column is a media type (image/video/audio)
      if (colType === "media" || colType === "image") {
        // Try multiple patterns for media tags
        if (typeof cell === "string") {
          // Pattern 1: <media id=xxx> (v0.2.0+ uses database ID)
          let match = cell.match(/<media id=(\d+)>/);
          // Pattern 2: &lt;media id=xxx&gt; (HTML escaped)
          if (!match) {
            match = cell.match(/&lt;media id=(\d+)&gt;/);
          }

          if (match) {
            const mediaId = match[1]; // Database ID (integer)
            // NEW: Use media ID resolution endpoint
            const url = `/api/boards/${experimentId.value}/media/by-id/${mediaId}`;
            console.log(
              `✅ Converted media tag to URL: ${url} (ID: ${mediaId})`,
            );
            return url;
          } else {
            console.log(`❌ No media tag match for: ${cell.substring(0, 50)}`);
          }
        }
      }
      return cell;
    });
  });

  return {
    ...currentTable.value,
    rows: processedRows,
  };
});
</script>

<template>
  <div class="table-viewer flex flex-col" :style="{ height: `${height}px` }">
    <div v-if="processedTableData" class="flex flex-col h-full">
      <div class="flex-1 overflow-auto relative">
        <!-- Shift indicator -->
        <div
          v-if="isShiftPressed"
          class="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-xs font-bold z-10"
        >
          SYNC MODE
        </div>
        <el-table :data="processedTableData.rows" size="small" stripe>
          <el-table-column
            v-for="(col, idx) in processedTableData.columns"
            :key="idx"
            :prop="String(idx)"
            :label="col"
            :width="
              processedTableData.column_types[idx] === 'image' ? 100 : undefined
            "
          >
            <template #default="{ row }">
              <img
                v-if="
                  processedTableData.column_types[idx] === 'media' ||
                  processedTableData.column_types[idx] === 'image'
                "
                :src="row[idx]"
                class="w-16 h-16 object-cover rounded"
                :alt="`${col} image`"
                @error="
                  (e) => {
                    console.error('Image load error:', row[idx]);
                    e.target.style.display = 'none';
                  }
                "
              />
              <span
                v-else-if="processedTableData.column_types[idx] === 'number'"
              >
                {{
                  typeof row[idx] === "number" ? row[idx].toFixed(3) : row[idx]
                }}
              </span>
              <span v-else>{{ row[idx] }}</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
      <div class="mt-2 flex justify-center">
        <div class="w-1/2">
          <div
            class="text-sm text-gray-600 dark:text-gray-400 mb-2 text-center"
          >
            Step: {{ processedTableData.step }}
            <span v-if="isShiftPressed" class="text-blue-500 font-bold ml-2">
              (Shift pressed - syncing all sliders)
            </span>
          </div>
          <el-slider
            v-model="stepIndex"
            :min="0"
            :max="tableData.length - 1"
            :marks="{ 0: 'Start', [tableData.length - 1]: 'End' }"
            :format-tooltip="
              (index) => `Step: ${tableData[index]?.step ?? index}`
            "
          />
        </div>
      </div>
    </div>
    <el-empty v-else description="No table data" />
  </div>
</template>
