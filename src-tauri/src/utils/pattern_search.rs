pub fn find_bytes(data: &[u8], pattern: &[u8]) -> Option<usize> {
    let n = data.len();
    let m = pattern.len();
    if m == 0 || n < m {
        return None;
    }

    let mut skip = [m; 256];
    for i in 0..m - 1 {
        skip[pattern[i] as usize] = m - 1 - i;
    }

    let mut i = 0;
    while i <= n - m {
        let mut j = m - 1;
        loop {
            if data[i + j] != pattern[j] {
                break;
            }
            if j == 0 {
                return Some(i);
            }
            j -= 1;
        }
        i += skip[data[i + m - 1] as usize];
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_bytes() {
        let data = b"mscorlib.dll\x00";
        assert_eq!(find_bytes(data, b"mscorlib.dll"), Some(0));
    }
}
