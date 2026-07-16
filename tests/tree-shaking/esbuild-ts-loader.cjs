'use strict';
const esbuild = require('esbuild');

module.exports = function(source) {
  const options = this.getOptions() || {};
  const result = esbuild.transformSync(source, {
    loader: 'ts',
    target: options.target || 'esnext',
  });
  return result.code;
};
