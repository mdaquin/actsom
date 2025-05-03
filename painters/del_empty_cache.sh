cd ldcache
rm `du * | grep '^4\s' | sed 's/4\s*\(..*\.json\)/\1/g'`
