<template>
  <div class="min-h-[calc(100vh-16rem)] flex items-center justify-center">
    <div
      class="w-full max-w-md bg-white dark:bg-gray-900 rounded-lg shadow-md p-8 border border-gray-200 dark:border-gray-800"
    >
      <h1
        class="text-2xl font-bold mb-6 text-center text-gray-900 dark:text-gray-100"
      >
        Login to KohakuBoard
      </h1>

      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-position="top"
        @submit.prevent="handleSubmit"
      >
        <el-form-item label="Username" prop="username">
          <el-input
            v-model="form.username"
            placeholder="Enter your username"
            size="large"
          />
        </el-form-item>

        <el-form-item label="Password" prop="password">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="Enter your password"
            size="large"
            show-password
          />
        </el-form-item>

        <el-button
          type="primary"
          size="large"
          class="w-full"
          :loading="loading"
          @click="handleSubmit"
        >
          Login
        </el-button>
      </el-form>

      <div class="mt-4 text-center text-sm text-gray-600 dark:text-gray-400">
        Don't have an account?
        <router-link
          to="/register"
          class="text-blue-500 dark:text-blue-400 hover:underline"
        >
          Sign up
        </router-link>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useAuthStore } from "@/stores/auth";
import { ElMessage } from "element-plus";

const router = useRouter();
const authStore = useAuthStore();
const formRef = ref(null);
const loading = ref(false);

const form = reactive({
  username: "",
  password: "",
});

const rules = {
  username: [
    { required: true, message: "Please enter username", trigger: "blur" },
  ],
  password: [
    { required: true, message: "Please enter password", trigger: "blur" },
  ],
};

async function handleSubmit() {
  if (!formRef.value) return;

  await formRef.value.validate(async (valid) => {
    if (!valid) return;

    loading.value = true;
    try {
      await authStore.login(form);
      ElMessage.success("Login successful");
      router.push("/");
    } catch (err) {
      ElMessage.error(err.response?.data?.detail || "Login failed");
    } finally {
      loading.value = false;
    }
  });
}
</script>
