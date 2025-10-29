<script setup>
import { useSliderSync } from "@/composables/useSliderSync";
import { ElMessage } from "element-plus";

const props = defineProps({
  mediaData: Array,
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
    if (!props.mediaData || props.mediaData.length === 0) return 0;
    const index = props.mediaData.findIndex(
      (item) => item.step === props.currentStep,
    );
    // If not found and autoAdvance enabled, return latest
    if (index === -1 && props.autoAdvanceToLatest) {
      return props.mediaData.length - 1;
    }
    return index >= 0 ? index : 0;
  },
  set(index) {
    if (props.mediaData && props.mediaData[index]) {
      const newStep = props.mediaData[index].step;
      emit("update:currentStep", newStep);

      // Check if user moved to latest - enable auto-advance
      if (index === props.mediaData.length - 1) {
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
  () => props.mediaData?.length,
  (newLength, oldLength) => {
    if (props.autoAdvanceToLatest && newLength > oldLength && newLength > 0) {
      // New data and auto-advance enabled - move to latest
      const latestStep = props.mediaData[newLength - 1].step;
      emit("update:currentStep", latestStep);
    }
  },
);

const currentMedia = computed(() => {
  if (!props.mediaData || props.mediaData.length === 0) return null;
  const index = stepIndex.value >= 0 ? stepIndex.value : 0;
  return props.mediaData[index];
});

// Setup slider synchronization
const { isShiftPressed, triggerSync } = useSliderSync(
  computed(() => `media-${props.cardId}`),
  computed(() => props.mediaData || []),
  (newStep) => {
    emit("update:currentStep", newStep);
  },
);

// Copy to clipboard function
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    ElMessage.success("Copied to clipboard");
  } catch (err) {
    ElMessage.error("Failed to copy");
  }
}
</script>

<template>
  <div class="media-viewer flex flex-col" :style="{ height: `${height}px` }">
    <div v-if="currentMedia" class="flex flex-col h-full gap-2">
      <!-- Media Display -->
      <div
        class="flex-1 flex items-center justify-center bg-gray-100 dark:bg-gray-800 rounded overflow-hidden relative"
      >
        <img
          v-if="currentMedia.type === 'image'"
          :src="currentMedia.url"
          :alt="currentMedia.caption"
          class="max-w-full max-h-full object-contain"
        />
        <video
          v-else-if="currentMedia.type === 'video'"
          :src="currentMedia.url"
          controls
          class="max-w-full max-h-full"
        />

        <!-- Shift indicator -->
        <div
          v-if="isShiftPressed"
          class="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-xs font-bold"
        >
          SYNC MODE
        </div>
      </div>

      <!-- Media Info -->
      <div class="px-2 space-y-1 text-sm">
        <!-- Name -->
        <div class="flex items-center gap-2">
          <span class="font-semibold text-gray-700 dark:text-gray-300 min-w-12"
            >Name:</span
          >
          <span
            class="flex-1 text-gray-900 dark:text-gray-100 truncate"
            :title="currentMedia.name"
          >
            {{ currentMedia.name || "Untitled" }}
          </span>
          <el-button
            size="small"
            text
            @click="copyToClipboard(currentMedia.name || '')"
            title="Copy name"
          >
            <i class="i-ep-document-copy"></i>
          </el-button>
        </div>

        <!-- Type -->
        <div class="flex items-center gap-2">
          <span class="font-semibold text-gray-700 dark:text-gray-300 min-w-12"
            >Type:</span
          >
          <span class="flex-1 text-gray-900 dark:text-gray-100">
            {{ currentMedia.type }}
          </span>
          <el-button
            size="small"
            text
            @click="copyToClipboard(currentMedia.type)"
            title="Copy type"
          >
            <i class="i-ep-document-copy"></i>
          </el-button>
        </div>

        <!-- URL -->
        <div class="flex items-center gap-2">
          <span class="font-semibold text-gray-700 dark:text-gray-300 min-w-12"
            >URL:</span
          >
          <a
            :href="currentMedia.url"
            target="_blank"
            class="flex-1 text-blue-500 dark:text-blue-400 hover:underline truncate"
            :title="currentMedia.url"
          >
            {{ currentMedia.url }}
          </a>
          <el-button
            size="small"
            text
            @click="copyToClipboard(currentMedia.url)"
            title="Copy URL"
          >
            <i class="i-ep-document-copy"></i>
          </el-button>
        </div>

        <!-- Caption -->
        <div v-if="currentMedia.caption" class="flex items-start gap-2">
          <span class="font-semibold text-gray-700 dark:text-gray-300 min-w-12"
            >Caption:</span
          >
          <p class="flex-1 text-gray-900 dark:text-gray-100">
            {{ currentMedia.caption }}
          </p>
          <el-button
            size="small"
            text
            @click="copyToClipboard(currentMedia.caption)"
            title="Copy caption"
          >
            <i class="i-ep-document-copy"></i>
          </el-button>
        </div>
      </div>

      <!-- Slider -->
      <div class="mt-2 flex justify-center px-2">
        <div class="w-full max-w-md">
          <div
            class="text-sm text-gray-600 dark:text-gray-400 mb-2 text-center"
          >
            Step: {{ currentMedia.step }}
            <span v-if="isShiftPressed" class="text-blue-500 font-bold ml-2">
              (Shift pressed - syncing all sliders)
            </span>
          </div>
          <el-slider
            v-model="stepIndex"
            :min="0"
            :max="mediaData.length - 1"
            :marks="{ 0: 'Start', [mediaData.length - 1]: 'End' }"
            :format-tooltip="
              (index) => `Step: ${mediaData[index]?.step ?? index}`
            "
          />
        </div>
      </div>
    </div>
    <el-empty v-else description="No media data" />
  </div>
</template>

<style scoped>
.media-viewer {
  display: flex;
  flex-direction: column;
}
</style>
