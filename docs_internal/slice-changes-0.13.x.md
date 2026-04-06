*adding this here so our docs don't auto-publish until 0.13.x is ready*

# docs/v1.3.x/guides/slicing-indexing.mdx

## Ellipses

The string ``'...'`` expands to the number of ``':'`` slices needed to index all dimensions. In most cases, this means that the length of the expanded selection tuple is ndim. There may only be a single ellipsis present.

```typescript
const m = np.array([[
  [0, 1, 2, 3],
  [4, 5, 6, 7],
  [8, 9, 10, 11],
]]);
// shape [1, 3, 4]

// Slice just the last dimension
m.slice('...', '1:3').toArray();
// shape [1, 3, 2]
// [[[1, 2],
//   [5, 6],
//   [9, 10]]]
```

## newaxis

The string ``'newaxis'`` serves to expand the dimensions of the result by one unit-length dimension. The added dimension is the position of the newaxis object in the indices.

```typescript
const m = np.array([
  [0, 1, 2, 3],
  [4, 5, 6, 7],
  [8, 9, 10, 11],
]);
// shape [3, 4]

// Insert an axis
m.slice(':', 'newaxis', ':').toArray();
// shape [3, 1, 4]
// [[[1, 2]],
//  [[5, 6]],
//  [[9, 10]]]
```

# Changelog

* Added slicing support for '...' and 'newaxis' and integers.