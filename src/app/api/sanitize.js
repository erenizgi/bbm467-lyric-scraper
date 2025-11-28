function sanitize(str) {
  return str.replace(/[<>:"/\\|?*]+/g, '').replace(/\s+/g, '_').trim();
}

export default sanitize;