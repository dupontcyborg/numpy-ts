/**
 * Tree-shaking test fixture: Single function import
 * Expected: Smallest possible bundle - only zeros and its dependencies
 */
import { zeros } from '../../../src/index';

// Use the function to prevent it from being eliminated
const arr = zeros([2, 3]);
console.log(arr.shape);
export { arr };
