#!/usr/bin/env bash
set -euxo pipefail

# from https://lindat.mff.cuni.cz/repository/items/7fbbbd99-ae2d-4b91-8318-d996dbe34cbc
curl -o "ud-treebanks-v2.18.tgz" "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/handle/11234/1-6149/ud-treebanks-v2.18.tgz"
#-o "ud-documentation-v2.18.tgz" "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/handle/11234/1-6149/ud-documentation-v2.18.tgz" 
#-o "ud-tools-v2.18.tgz" "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/handle/11234/1-6149/ud-tools-v2.18.tgz"

tar -xvzf ud-treebanks-v2.18.tgz