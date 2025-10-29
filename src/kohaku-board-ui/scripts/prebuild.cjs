#!/usr/bin/env node
/**
 * Prebuild script for KohakuBoard frontend
 * Copies logo files from root images/ to public/images/
 * Copies documentation from docs/kohakuboard/ to public/docs/
 */

const fs = require('fs');
const path = require('path');

const rootDir = path.join(__dirname, '..', '..', '..');
const publicDir = path.join(__dirname, '..', 'public');
const imagesPublicDir = path.join(publicDir, 'images');
const docsPublicDir = path.join(publicDir, 'docs');

const logoFiles = [
  {
    source: path.join(rootDir, 'images', 'logo-square.svg'),
    dest: path.join(imagesPublicDir, 'logo-square.svg'),
  },
  {
    source: path.join(rootDir, 'images', 'logo-banner.svg'),
    dest: path.join(imagesPublicDir, 'logo-banner.svg'),
  },
  {
    source: path.join(rootDir, 'images', 'logo-banner-dark.svg'),
    dest: path.join(imagesPublicDir, 'logo-banner-dark.svg'),
  },
  {
    source: path.join(rootDir, 'images', 'logo-square.svg'),
    dest: path.join(publicDir, 'favicon.svg'),
  },
];

function copyFile(source, dest) {
  try {
    const destDir = path.dirname(dest);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }
    fs.copyFileSync(source, dest);
    console.log(`  ✓ ${path.relative(rootDir, source)} → ${path.relative(publicDir, dest)}`);
  } catch (error) {
    console.error(`  ✗ Failed to copy ${source}: ${error.message}`);
    process.exit(1);
  }
}

function copyDirectory(source, dest) {
  try {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }

    const entries = fs.readdirSync(source, { withFileTypes: true });

    for (const entry of entries) {
      const sourcePath = path.join(source, entry.name);
      const destPath = path.join(dest, entry.name);

      if (entry.isDirectory()) {
        copyDirectory(sourcePath, destPath);
      } else {
        fs.copyFileSync(sourcePath, destPath);
        console.log(`  ✓ ${path.relative(rootDir, sourcePath)} → ${path.relative(publicDir, destPath)}`);
      }
    }
  } catch (error) {
    console.error(`  ✗ Failed to copy directory ${source}: ${error.message}`);
    process.exit(1);
  }
}

function main() {
  console.log('📦 KohakuBoard Prebuild\n');

  // Copy logo files
  console.log('📚 Copying logo files...');
  if (!fs.existsSync(imagesPublicDir)) {
    fs.mkdirSync(imagesPublicDir, { recursive: true });
  }
  logoFiles.forEach(file => copyFile(file.source, file.dest));

  // Copy documentation
  console.log('\n📖 Copying documentation...');
  const docsSource = path.join(rootDir, 'docs', 'kohakuboard');
  if (fs.existsSync(docsSource)) {
    copyDirectory(docsSource, docsPublicDir);
  } else {
    console.log('  ⚠️  Documentation not found (skipping): docs/kohakuboard/');
  }

  console.log('\n✅ Prebuild completed successfully!');
}

main();
